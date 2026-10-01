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
from torch.utils.checkpoint import checkpoint


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

    def __init__(self, dim, heads, block_size, rope, ffn_ratio, query_chunk_size,
                 checkpoint_chunks=False):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.block_size, self.query_chunk_size = block_size, query_chunk_size
        self.rope = rope
        self.checkpoint_chunks = checkpoint_chunks
        self.attn_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)

    def forward_chunk(self, x, previous=None, start=0):
        """Encode a block-aligned chunk, using the preceding layer-input block.

        Only the final chunk may have a partial block. This is also the
        primitive for sequence-parallel halos and out-of-core prefill.
        Absolute positions are used, including after the 2**24 boundary.
        """
        batch, n, dim = x.shape
        b = self.block_size
        groups = (n + b - 1) // b
        if start % b:
            raise ValueError("halo chunk start must be block aligned")
        if previous is None:
            previous = x.new_zeros(batch, b, dim)
            valid_start = start
        else:
            if previous.shape != (batch, b, dim):
                raise ValueError("previous halo must contain exactly one block")
            valid_start = max(0, start - b)
        xs = torch.cat((previous, F.pad(x, (0, 0, 0, groups * b - n))), 1)
        q, k, v = self.qkv(self.attn_norm(xs)).reshape(
            batch, (groups + 1) * b, 3, self.heads, self.hd).unbind(2)
        q = q.reshape(batch, groups + 1, b, self.heads, self.hd)[:, 1:]
        k = k.reshape(batch, groups + 1, b, self.heads, self.hd)
        v = v.reshape(batch, groups + 1, b, self.heads, self.hd)
        kw = torch.cat((k[:, :-1], k[:, 1:]), 2)
        vw = torch.cat((v[:, :-1], v[:, 1:]), 2)
        # Use the query block's origin for BOTH q and k. Relative rotations
        # are unchanged, chunk boundaries cannot affect rounding, and fp32
        # absolute-position aliasing above 2**24 is avoided.
        q = self.rope(q, torch.arange(b, device=x.device))
        kw = self.rope(kw, torch.arange(-b, b, device=x.device))
        offs_q = torch.arange(b, device=x.device)
        offs_k = torch.arange(2 * b, device=x.device)
        causal = offs_k[None, :] <= b + offs_q[:, None]
        kp = start + torch.arange(groups, device=x.device)[:, None, None] * b - b + offs_k[None, None, :]
        mask = causal[None] & (kp >= valid_start) & (kp < start + n)
        context = _attention(q, kw, vw, mask).reshape(batch, groups * b, dim)[:, :n]
        y = x + self.out(context)
        return y + self.ffn(self.ffn_norm(y))

    def forward(self, x, positions=None, previous=None, start=0):
        if positions is not None:
            # Callers use contiguous, absolute token positions.
            start = int(positions[0])
        step = self.query_chunk_size * self.block_size
        parts = []
        result = torch.empty_like(x) if not torch.is_grad_enabled() else None
        for lo in range(0, x.shape[1], step):
            hi = min(lo + step, x.shape[1])
            prev = previous if lo == 0 else x[:, lo-self.block_size:lo]
            if self.checkpoint_chunks and torch.is_grad_enabled():
                y = checkpoint(self.forward_chunk, x[:, lo:hi], prev, start + lo,
                               use_reentrant=False, preserve_rng_state=False)
            else:
                y = self.forward_chunk(x[:, lo:hi], prev, start + lo)
            if result is None:
                parts.append(y)
            else:
                result[:, lo:hi] = y
        return torch.cat(parts, 1) if result is None else result


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
