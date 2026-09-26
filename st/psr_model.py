"""Pair-state readout (PSR) transformer: bilateral content channel.

Standard attention is asymmetric. The addressing channel is bilateral — both
sides are projected (q_t . k_j) — while the content channel is unilateral:
the edge (t, j) transmits alpha_tj * v_j, a scalar times a fixed vector, and
the reader contributes nothing but the scalar. PSR restores symmetry on the
content channel:

    c_j = gelu(W_c h_j)     # content factor; nonlinearity at WRITE time, so
                            # the weighted sum below is a mean embedding of
                            # the read set, not a bare first moment
    p_t = W_p h_t           # reader-side content projection (zero-init
                            # weight, ones-init bias: starts as all-ones)
    m_t = sum_j alpha_tj c_j    # the usual AV pass, value width r_c
    o_t = W_o (p_t * m_t)   # pair-state modulation BEFORE the output proj

The edge now carries r_c numbers of pair-dependent content instead of one
scalar times a reader-independent vector; the pair state p_t * c_j is never
materialized per edge because p_t * sum(alpha c) == sum(alpha (p_t * c_j)),
so the whole change is: wider V with write-time GELU, one elementwise
multiply by the reader's own projection, wider W_o. Everything runs through
F.scaled_dot_product_attention with is_causal=True (value dim may differ
from qk dim); there is no custom kernel and no n^2 score materialization.

Degenerate point: p_mode="ones" + write_gelu=False + r_c == dim//heads is
bit-exact standard causal attention, so one class provides both the
baseline and the treatment. RoPE lives on Q/K only, as usual; C and P carry
no position.
"""
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import RotaryEmbedding, FeedForward


class PSRBlock(nn.Module):
    """Pre-norm causal attention with pair-state readout + FFN."""

    def __init__(self, dim, heads, d_a, r_c, rope, ffn_ratio, write_gelu,
                 p_mode):
        super().__init__()
        self.heads, self.d_a, self.r_c = heads, d_a, r_c
        self.rope = rope
        self.write_gelu = write_gelu
        self.attn_norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, heads * d_a, bias=False)
        self.wk = nn.Linear(dim, heads * d_a, bias=False)
        self.wc = nn.Linear(dim, heads * r_c, bias=False)
        if p_mode == "learned":
            self.wp = nn.Linear(dim, heads * r_c, bias=True)
            nn.init.zeros_(self.wp.weight)
            nn.init.ones_(self.wp.bias)
        else:  # "ones": reader does not modulate; no parameter
            self.wp = None
        self.out = nn.Linear(heads * r_c, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)

    def forward(self, x, positions):
        batch, n, _ = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(batch, n, self.heads, self.d_a)
        k = self.wk(h).view(batch, n, self.heads, self.d_a)
        c = self.wc(h).view(batch, n, self.heads, self.r_c)
        if self.write_gelu:
            c = F.gelu(c)
        q, k = self.rope(q, positions), self.rope(k, positions)
        m = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), c.transpose(1, 2),
            is_causal=True).transpose(1, 2)             # [B,N,H,r_c]
        if self.wp is not None:
            p = self.wp(h).view(batch, n, self.heads, self.r_c)
            m = p * m
        y = x + self.out(m.reshape(batch, n, self.heads * self.r_c))
        return y + self.ffn(self.ffn_norm(y))


class PSRModel(nn.Module):
    """forward([B,N], sup=None) -> [B,N,vocab_size].

    sup is accepted for signature compatibility with the stack training
    driver and ignored: dense LM supervision covers all positions.
    """

    def __init__(self, vocab_size, dim=256, heads=4, layers=6, d_a=None,
                 r_c=None, write_gelu=True, p_mode="learned", ffn_ratio=4):
        super().__init__()
        d_a = d_a if d_a is not None else dim // heads
        r_c = r_c if r_c is not None else dim // heads
        integers = {"vocab_size": vocab_size, "dim": dim, "heads": heads,
                    "layers": layers, "d_a": d_a, "r_c": r_c}
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if d_a % 2:
            raise ValueError("d_a must be even for RoPE")
        if p_mode not in ("learned", "ones"):
            raise ValueError("p_mode must be 'learned' or 'ones'")
        if not math.isfinite(ffn_ratio) or ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be finite and positive")
        self.dim, self.heads, self.d_a, self.r_c = dim, heads, d_a, r_c
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rope = RotaryEmbedding(d_a)
        self.blocks = nn.ModuleList(
            PSRBlock(dim, heads, d_a, r_c, self.rope, ffn_ratio, write_gelu,
                     p_mode)
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
