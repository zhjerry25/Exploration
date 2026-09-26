"""Shared building blocks for the stack model: local halo attention,
RoPE, FFN and raw KV projection.

The halo block runs causal attention over the previous block ++ the current
block (2b window) with blocks folded into the batch/group axes; chunking
bounds temporary score memory, not total training activations retained by
autograd.
"""
import torch
from torch import nn
from torch.nn import functional as F


FOLD_CHUNK = 8192


def _pad_sequence(x, length):
    """Right-pad axis 1, leaving all feature axes unchanged."""
    if x.shape[1] == length:
        return x
    return torch.cat([x, x.new_zeros(x.shape[0], length - x.shape[1], *x.shape[2:])], dim=1)


def _gather(source, indices):
    """[B,N,H,Dh] and [B,G,K] -> [B,G,K,H,Dh]."""
    rows = torch.arange(source.shape[0], device=source.device)[:, None, None]
    return source[rows, indices]


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
    """Pre-norm causal attention over [prev block ++ current block] + FFN."""

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
        q, k, v = self.qkv(self.attn_norm(x)).reshape(
            batch, n, 3, self.heads, self.hd
        ).unbind(2)
        q, k = self.rope(q, positions), self.rope(k, positions)
        q = _pad_sequence(q, groups * b).reshape(batch, groups, b, self.heads, self.hd)
        offsets = torch.arange(b, device=x.device)
        window = torch.arange(-b, b, device=x.device)
        outputs = []
        for start in range(0, groups, self.query_chunk_size):
            stop = min(groups, start + self.query_chunk_size)
            base = torch.arange(start, stop, device=x.device) * b
            qpos = base[:, None] + offsets
            kpos = base[:, None] + window
            valid = (kpos >= 0) & (kpos < n)
            mask = valid[:, None, :] & (kpos[:, None, :] <= qpos[:, :, None])
            idx = kpos.clamp(0, n - 1).unsqueeze(0).expand(batch, -1, -1)
            outputs.append(_attention(q[:, start:stop], _gather(k, idx), _gather(v, idx),
                                      mask.unsqueeze(0)))
        context = torch.cat(outputs, dim=1).reshape(batch, groups * b, dim)[:, :n]
        y = x + self.out(context)
        return y + self.ffn(self.ffn_norm(y))


class KVProjection(nn.Module):
    def __init__(self, dim, heads, rope, use_rope=True):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 2 * dim, bias=False)
        self.rope = rope
        self.use_rope = use_rope

    def forward(self, x, positions):
        k, v = self.proj(self.norm(x)).reshape(
            x.shape[0], x.shape[1], 2, self.heads, self.hd
        ).unbind(2)
        if self.use_rope:
            k = self.rope(k, positions)
        return k, v
