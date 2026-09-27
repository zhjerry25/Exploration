"""Dense causal transformer baseline, parameter- and depth-matched to StackModel.

Same embedding/FFN shapes and the same driver interface as StackModel:
forward([B,N], sup=None) -> [B,N,vocab_size] with zero logits at unsupervised
positions. Blocks are standard pre-norm causal attention (RoPE) + FFN, so a
`--layers 3` baseline matches `arch="Lx2,G"` in both parameter count and
layer passes (3 x 12d^2 vs 24d^2 local + 2d^2 raw_kv + 10d^2 read).
"""
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import RotaryEmbedding, FeedForward


class BaselineBlock(nn.Module):
    """Pre-norm full causal attention + FFN (same shapes as HaloMemoryBlock)."""

    def __init__(self, dim, heads, rope, ffn_ratio):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.rope = rope
        self.attn_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)

    def forward(self, x, positions):
        batch, n, dim = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).reshape(
            batch, n, 3, self.heads, self.hd
        ).unbind(2)
        q, k = self.rope(q, positions), self.rope(k, positions)
        ctx = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True)
        y = x + self.out(ctx.transpose(1, 2).reshape(batch, n, dim))
        return y + self.ffn(self.ffn_norm(y))


class BaselineModel(nn.Module):
    """forward([B,N], sup=None) -> [B,N,vocab_size]."""

    def __init__(self, vocab_size, dim=256, heads=4, layers=3, ffn_ratio=4):
        super().__init__()
        for name, value in {"vocab_size": vocab_size, "dim": dim,
                            "heads": heads, "layers": layers}.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if dim % heads or (dim // heads) % 2:
            raise ValueError("dim / heads must be an even integer for RoPE")
        if not math.isfinite(ffn_ratio) or ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be finite and positive")
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rope = RotaryEmbedding(self.hd)
        self.blocks = nn.ModuleList(
            BaselineBlock(dim, heads, self.rope, ffn_ratio)
            for _ in range(layers))
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, sup=None):
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        x = self.embedding(input_ids)
        positions = torch.arange(input_ids.shape[1], device=x.device)
        for block in self.blocks:
            x = block(x, positions)
        logits = self.lm_head(self.final_norm(x))
        if sup is not None:
            logits = logits * sup.unsqueeze(-1)
        return logits
