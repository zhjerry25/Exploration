"""Shared building blocks for the stack model: local halo attention,
RoPE, FFN and raw KV projection.

Everything is streamed in query chunks: qkv projection, RoPE, the halo
attention itself and the FFN all run on slices of a few blocks, so peak
memory is independent of sequence length (long-context eval is bounded by
the stored x / raw K/V, not by width-3d/4d temporaries). Chunk size never
changes results (guarded by tests); folding bounds temporary score memory
for CUDA backends with grid-size limits.
"""
import torch
from torch import nn
from torch.nn import functional as F


FOLD_CHUNK = 8192


def _gather_heads(source, indices):
    """[B,H,S,Dh] and [B,H,G,K] -> [B,H,G,K,Dh] (gather along S)."""
    dh = source.shape[-1]
    src = source.unsqueeze(3).expand(-1, -1, -1, indices.shape[3], -1)
    return src.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, -1, dh))


def _attention(q, k, v, mask):
    """SDPA for [B,G,T,H,Dh], with a boolean [B or 1,G,T,K] mask.

    Callers provide at least one visible key per row (including padded queries).
    Folded rows are bounded for CUDA backends with grid-size limits.
    """
    batch, groups, length, heads, hd = q.shape
    q = q.permute(0, 1, 3, 2, 4).reshape(batch * groups, heads, length, hd)
    k = k.permute(0, 1, 3, 2, 4).reshape(batch * groups, heads, -1, hd)
    v = v.permute(0, 1, 3, 2, 4).reshape(batch * groups, heads, -1, hd)
    mask = mask.expand(batch, groups, length, k.shape[2]).reshape(
        batch * groups, 1, length, k.shape[2]
    )
    parts = [F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mi)
             for qi, ki, vi, mi in zip(q.split(FOLD_CHUNK), k.split(FOLD_CHUNK),
                                      v.split(FOLD_CHUNK), mask.split(FOLD_CHUNK))]
    return torch.cat(parts, dim=0).transpose(1, 2).reshape(batch, groups, length, heads, hd)


class RotaryEmbedding(nn.Module):
    """Split-half RoPE at original-token coordinates, with no length table."""

    def __init__(self, head_dim, base=10_000.0):
        super().__init__()
        self.head_dim, self.base = head_dim, base
        # Integer indices survive model.to(bfloat16) without rounding frequencies.
        self.register_buffer("frequency_index", torch.arange(0, head_dim, 2), persistent=False)

    def forward(self, x, positions):
        # x: [..., H, Dh]; positions broadcast to x's axes before H.
        with torch.autocast(device_type=x.device.type, enabled=False):
            inv = self.base ** (-self.frequency_index.float() / self.head_dim)
            angles = positions.float().unsqueeze(-1) * inv
            cos = angles.cos().unsqueeze(-2).to(x.dtype)
            sin = angles.sin().unsqueeze(-2).to(x.dtype)
        first, second = x.chunk(2, dim=-1)
        return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


class FeedForward(nn.Sequential):
    def __init__(self, dim, ratio):
        hidden = max(1, round(dim * ratio))
        super().__init__(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class HaloMemoryBlock(nn.Module):
    """Pre-norm causal attention over [prev block ++ current block] + FFN,
    fully streamed per query chunk (peak memory independent of n)."""

    def __init__(self, dim, heads, block_size, rope, ffn_ratio, query_chunk_size):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.block_size, self.query_chunk_size = block_size, query_chunk_size
        self.rope = rope
        self.attn_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)

    def forward(self, x, positions):
        batch, n, dim = x.shape
        b = self.block_size
        groups = (n + b - 1) // b
        offs_q = torch.arange(b, device=x.device)
        offs_k = torch.arange(2 * b, device=x.device)
        # window entry u of query group g is token (g-1)*b+u; causal: u <= b+w
        causal = offs_k[None, :] <= b + offs_q[:, None]          # [b,2b]
        outputs = []
        for start in range(0, groups, self.query_chunk_size):
            stop = min(groups, start + self.query_chunk_size)
            gc = stop - start
            lo = (start - 1) * b                    # previous block for K/V
            seg = x[:, max(lo, 0):min(stop * b, n)]
            left = b if start == 0 else 0           # virtual zero block
            xs = F.pad(seg, (0, 0, left, (gc + 1) * b - left - seg.shape[1]))
            pos_s = torch.arange(lo, lo + (gc + 1) * b, device=x.device)
            q, k, v = self.qkv(self.attn_norm(xs)).reshape(
                batch, (gc + 1) * b, 3, self.heads, self.hd).unbind(2)
            q, k = self.rope(q, pos_s), self.rope(k, pos_s)
            q = q.reshape(batch, gc + 1, b, self.heads, self.hd)[:, 1:]
            k = k.reshape(batch, gc + 1, b, self.heads, self.hd)
            v = v.reshape(batch, gc + 1, b, self.heads, self.hd)
            k_w = torch.cat([k[:, :-1], k[:, 1:]], dim=2)  # [B,gc,2b,H,hd]
            v_w = torch.cat([v[:, :-1], v[:, 1:]], dim=2)
            key_pos = (torch.arange(start, stop, device=x.device)[:, None, None]
                       * b - b + offs_k[None, None, :])    # [gc,1,2b] global
            valid = (key_pos >= 0) & (key_pos < n)
            mask = causal[None] & valid                    # [gc,b,2b]
            outputs.append(_attention(q, k_w, v_w, mask))
        context = torch.cat(outputs, dim=1).reshape(batch, groups * b, dim)[:, :n]
        y = x + self.out(context)
        zs = []
        step = self.query_chunk_size * b
        for s0 in range(0, n, step):                # FFN is position-wise
            yc = y[:, s0:s0 + step]
            zs.append(yc + self.ffn(self.ffn_norm(yc)))
        return torch.cat(zs, dim=1)


class KVProjection(nn.Module):
    """LayerNorm + linear -> (k, v), projected in token chunks so the
    width-2d transient is bounded at long n (chunking never changes values)."""

    def __init__(self, dim, heads, rope, use_rope=True, chunk_tokens=1 << 16):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 2 * dim, bias=False)
        self.rope = rope
        self.use_rope = use_rope
        self.chunk_tokens = chunk_tokens

    def forward(self, x, positions):
        n = x.shape[1]
        ks, vs = [], []
        for s0 in range(0, n, self.chunk_tokens):
            sl = slice(s0, s0 + self.chunk_tokens)
            k, v = self.proj(self.norm(x[:, sl])).reshape(
                x.shape[0], -1, 2, self.heads, self.hd
            ).unbind(2)
            if self.use_rope:
                k = self.rope(k, positions[sl])
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, 1), torch.cat(vs, 1)
