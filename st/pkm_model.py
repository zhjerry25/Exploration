"""Product-key memory (PKM) transformer: the FFN's addressing made explicit.

A standard FFN is already a table: W_1's rows are keys, W_2's rows are
values, and the query h addresses them by a dense linear scan — every token
scores every row. Row count and scan cost are locked together, which is why
table growth by widening/depth has flat marginal param efficiency.

PKM decouples them. The query (P heads, d_k dims each) is split in two; the
row keys live in two sub-key sets K1, K2 (sqrt(K) each) whose Cartesian
product is the full table:

    s = q1 . K1 + q2 . K2,   top-m per set, m^2 cells, softmax over cells
    out = W_o ( mean_h sum_cells w * U_row )

Addressing costs O(P * sqrt(K) * d_k) and the read gathers only m^2 rows per
head, so the table can grow from 2.4k to 16k rows at ~constant per-token
FLOPs. The attention half of the block is untouched standard pre-norm
causal attention.

Two failure-mode instruments are built in: `last_idx` (the selected row
ids) feeds utilization/usage-entropy logs — sparse tables can collapse
(rich-get-richer, most rows starve) — and an optional sub-key uniformity
aux loss (`balance`) is the counter-pressure. Row usage is also the purity
probe: rows are only useful when their input distribution is narrow.
"""
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import RotaryEmbedding


class PKMTable(nn.Module):
    """Product-key addressed row table (drop-in FFN replacement)."""

    def __init__(self, dim, rows, m, heads, balance, ckpt=False):
        super().__init__()
        S = math.isqrt(rows)
        if S * S != rows:
            raise ValueError("pkm_k must be a perfect square")
        if S < m:
            raise ValueError("sqrt(pkm_k) must be >= pkm_m")
        self.rows, self.S, self.m, self.heads = rows, S, m, heads
        self.ckpt = ckpt
        self.dk = dim // heads
        if self.dk % 2:
            raise ValueError("dim // pkm_heads must be even")
        self.balance = balance
        self.wq = nn.Linear(dim, heads * self.dk, bias=False)
        self.k1 = nn.Parameter(torch.randn(heads, S, self.dk // 2) * 0.02)
        self.k2 = nn.Parameter(torch.randn(heads, S, self.dk // 2) * 0.02)
        self.values = nn.Parameter(torch.randn(rows, dim) * 0.02)
        self.out = nn.Linear(dim, dim, bias=False)
        self.last_idx = None  # detached probe: [B,N,H,m*m] row ids

    def forward(self, h):
        if self.ckpt and self.training and h.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward, h, use_reentrant=False)
        return self._forward(h)

    def _forward(self, h):
        batch, n, dim = h.shape
        q = self.wq(h).view(batch, n, self.heads, self.dk)
        with torch.autocast(device_type=h.device.type, enabled=False):
            # score in fp32 under low-precision autocast, else keep dtype
            dt = (torch.float32 if h.dtype in (torch.bfloat16, torch.float16)
                  else h.dtype)
            q1, q2 = q.to(dt).chunk(2, -1)
            s1 = torch.einsum('bnhk,hsk->bnhs', q1, self.k1.to(dt))
            s2 = torch.einsum('bnhk,hsk->bnhs', q2, self.k2.to(dt))
            v1, i1 = s1.topk(self.m, dim=-1)        # [B,N,H,m]
            v2, i2 = s2.topk(self.m, dim=-1)
            cells = v1[..., None] + v2[..., None, :]     # [B,N,H,m,m]
            w = cells.flatten(-2).softmax(-1)            # [B,N,H,m*m]
            aux = None
            if self.balance:
                p1 = s1.softmax(-1).mean((0, 1, 2))
                p2 = s2.softmax(-1).mean((0, 1, 2))
                aux = self.S * ((p1 * p1).sum() + (p2 * p2).sum())
        idx = (i1[..., None] * self.S + i2[..., None, :]).flatten(-2)
        self.last_idx = idx.detach()
        vals = self.values[idx]                          # [B,N,H,m*m,dim]
        read = (w.to(vals.dtype)[..., None] * vals).sum(-2)
        return self.out(read.mean(2)), aux


class PKMBlock(nn.Module):
    """Pre-norm causal attention + PKM table."""

    def __init__(self, dim, heads, rope, rows, m, pkm_heads, balance, ckpt):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.rope = rope
        self.attn_norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.table = PKMTable(dim, rows, m, pkm_heads, balance, ckpt)

    def forward(self, x, positions):
        batch, n, dim = x.shape
        h = self.attn_norm(x)
        q = self.rope(self.wq(h).view(batch, n, self.heads, self.hd),
                      positions)
        k = self.rope(self.wk(h).view(batch, n, self.heads, self.hd),
                      positions)
        v = self.wv(h).view(batch, n, self.heads, self.hd)
        m = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True).transpose(1, 2)
        y = x + self.out(m.reshape(batch, n, dim))
        z, aux = self.table(self.ffn_norm(y))
        return y + z, aux


class PKMModel(nn.Module):
    """forward([B,N], sup=None) -> [B,N,vocab_size]; aux loss on self.aux."""

    def __init__(self, vocab_size, dim=256, heads=4, layers=6, pkm_k=2401,
                 pkm_m=2, pkm_heads=4, balance=0.0, ckpt=False):
        super().__init__()
        integers = {"vocab_size": vocab_size, "dim": dim, "heads": heads,
                    "layers": layers, "pkm_k": pkm_k, "pkm_m": pkm_m,
                    "pkm_heads": pkm_heads}
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if (dim // heads) % 2:
            raise ValueError("dim / heads must be even for RoPE")
        if not math.isfinite(balance) or balance < 0:
            raise ValueError("balance must be finite and >= 0")
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.balance = balance
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rope = RotaryEmbedding(self.hd)
        self.blocks = nn.ModuleList(
            PKMBlock(dim, heads, self.rope, pkm_k, pkm_m, pkm_heads, balance,
                     ckpt)
            for _ in range(layers))
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.aux = None

    def forward(self, input_ids, sup=None):
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        n = input_ids.shape[1]
        x = self.embedding(input_ids)
        positions = torch.arange(n, device=x.device)
        auxes = []
        for block in self.blocks:
            x, aux = block(x, positions)
            if aux is not None:
                auxes.append(aux)
        self.aux = (self.balance * sum(auxes)) if auxes else None
        return self.lm_head(self.final_norm(x))
