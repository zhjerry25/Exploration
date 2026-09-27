"""Iterative-readout transformer (IRT): reader-dependence bought with
compute, not parameters.

The attention read in a standard block is single-shot: the query is written
before anything is read, so conditional retrieval ("what I just read tells
me where to look next") requires another whole layer. IRT instead iterates
the read side over a frozen per-block K/V cache:

    h  = LN(x);  K, V = W_k h, W_v h          # written once per block
    q0 = RoPE(W_q h);  m0 = SDPA(q0, K, V)
    q_r = q0 + U_r m_{r-1}                    # zero-init U_r: starts as q0
    m_r = SDPA(q_r, K, V)                     # re-score, re-read, same cache
    o  = W_o [m0; m_{T-1}]                    # first read ++ final read

Each extra round costs one d x d projection and one SDPA pass; K/V and the
FFN are amortized. Hop budget is layers x rounds: a 1-layer 2-round block
can do two-hop retrieval that a 1-layer single-read block cannot express at
all. At read_rounds=1 there is no U_r and W_o is d x d — bit-exact standard
causal attention, so one class provides both baseline and treatment.
"""
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import RotaryEmbedding, FeedForward


class IRTBlock(nn.Module):
    """Pre-norm causal attention with T read rounds over frozen K/V + FFN."""

    def __init__(self, dim, heads, read_rounds, rope, ffn_ratio):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.read_rounds = read_rounds
        self.rope = rope
        self.attn_norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        # query refinements for rounds 1..T-1; zero-init => round r starts as
        # an exact copy of round 0 (fresh model degenerates to T=1 behavior)
        self.u = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(read_rounds - 1))
        for u in self.u:
            nn.init.zeros_(u.weight)
        self.out = nn.Linear(dim * (2 if read_rounds > 1 else 1), dim,
                             bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)

    def _read(self, q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True).transpose(1, 2)  # [B,N,H,hd]

    def forward(self, x, positions):
        batch, n, dim = x.shape
        h = self.attn_norm(x)
        k = self.wk(h).view(batch, n, self.heads, self.hd)
        v = self.wv(h).view(batch, n, self.heads, self.hd)
        q0 = self.rope(self.wq(h).view(batch, n, self.heads, self.hd),
                       positions)
        k = self.rope(k, positions)
        m0 = self._read(q0, k, v)
        m = m0
        for u in self.u:
            q = q0 + u(m.reshape(batch, n, dim)).view(batch, n, self.heads,
                                                      self.hd)
            m = self._read(q, k, v)
        if self.u:  # [first read (all heads); final read (all heads)]
            m = torch.cat([m0.reshape(batch, n, dim),
                           m.reshape(batch, n, dim)], dim=-1)
            y = x + self.out(m)
        else:
            y = x + self.out(m.reshape(batch, n, dim))
        return y + self.ffn(self.ffn_norm(y))


class IRTModel(nn.Module):
    """forward([B,N], sup=None) -> [B,N,vocab_size].

    sup is accepted for signature compatibility with the training driver and
    ignored: dense LM supervision covers all positions.
    """

    def __init__(self, vocab_size, dim=256, heads=4, layers=6, read_rounds=2,
                 ffn_ratio=4):
        super().__init__()
        integers = {"vocab_size": vocab_size, "dim": dim, "heads": heads,
                    "layers": layers, "read_rounds": read_rounds}
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if (dim // heads) % 2:
            raise ValueError("dim / heads must be even for RoPE")
        if not math.isfinite(ffn_ratio) or ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be finite and positive")
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rope = RotaryEmbedding(self.hd)
        self.blocks = nn.ModuleList(
            IRTBlock(dim, heads, read_rounds, self.rope, ffn_ratio)
            for _ in range(layers))
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, sup=None):
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        n = input_ids.shape[1]
        x = self.embedding(input_ids)
        positions = torch.arange(n, device=x.device)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.final_norm(x))
