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
from torch.utils.checkpoint import checkpoint

from .blocks import RotaryEmbedding, FeedForward
from ..runtime.parallel import ParallelContext


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

    def forward(self, x, positions, parallel=None, checkpoint_chunks=False):
        parallel = parallel or ParallelContext()
        batch, n, dim = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).reshape(
            batch, n, 3, self.heads, self.hd
        ).unbind(2)
        q, k = self.rope(q, positions), self.rope(k, positions)
        q, k, v = (parallel.to_heads(t) for t in (q, k, v))
        ctx = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True)
        ctx = parallel.to_sequence(ctx.transpose(1, 2)).reshape(batch, n, dim)
        def finish(hidden, context):
            y = hidden + self.out(context)
            return y + self.ffn(self.ffn_norm(y))
        if checkpoint_chunks and torch.is_grad_enabled():
            return checkpoint(finish, x, ctx, use_reentrant=False, preserve_rng_state=False)
        return finish(x, ctx)


class BaselineModel(nn.Module):
    """forward([B,N], sup=None) -> [B,N,vocab_size]."""

    def __init__(self, vocab_size, dim=256, heads=4, layers=3, ffn_ratio=4,
                 checkpoint_chunks=False, loss_chunk=128):
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
        self.block_size = 1  # sequence-shard alignment; no local block semantics
        self.checkpoint_chunks, self.loss_chunk = checkpoint_chunks, loss_chunk
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rope = RotaryEmbedding(self.hd)
        self.blocks = nn.ModuleList(
            BaselineBlock(dim, heads, self.rope, ffn_ratio)
            for _ in range(layers))
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def encode(self, input_ids, parallel=None, offset=0):
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        x = self.embedding(input_ids)
        positions = torch.arange(offset, offset+input_ids.shape[1], device=x.device)
        for block in self.blocks:
            x = block(x, positions, parallel, self.checkpoint_chunks)
        return x

    def iter_logits(self, input_ids, positions):
        """Bound the vocabulary projection; the dense baseline encoder stays resident."""
        x = self.encode(input_ids)
        rows = torch.arange(x.shape[0], device=x.device)[:, None]
        for lo in range(0, positions.shape[1], self.loss_chunk):
            sl = slice(lo, min(lo+self.loss_chunk, positions.shape[1]))
            z = x[rows, positions[:, sl]]
            yield sl, self.lm_head(self.final_norm(z)), None

    def forward_loss(self, input_ids, targets, sup=None, parallel=None, offset=0):
        if input_ids.shape != targets.shape:
            raise ValueError("input_ids and targets must have the same shape")
        if self.loss_chunk < 1:
            raise ValueError("loss_chunk must be positive")
        if sup is not None and (sup.shape != targets.shape or sup.dtype != torch.bool):
            raise ValueError("sup must be bool [B,N]")
        valid = targets != -100
        if sup is not None:
            valid = valid & sup
        x = self.encode(input_ids, parallel, offset)
        indices = valid.nonzero()
        # Connected zero handles a rank with no supervised tokens without
        # omitting the baseline's collective backward or head parameters.
        loss = x.sum()*0 + self.final_norm.weight.sum()*0 + self.final_norm.bias.sum()*0 + self.lm_head.weight.sum()*0
        correct = torch.zeros((), device=x.device, dtype=torch.long)
        errors = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        def head_loss(hidden, labels):
            logits = self.lm_head(self.final_norm(hidden)).float()
            return F.cross_entropy(logits, labels, reduction="sum"), logits.argmax(-1).eq(labels)
        for lo in range(0, indices.shape[0], self.loss_chunk):
            ix = indices[lo:lo+self.loss_chunk]
            hidden, labels = x[ix[:, 0], ix[:, 1]], targets[ix[:, 0], ix[:, 1]]
            if self.checkpoint_chunks and torch.is_grad_enabled():
                part, hit = checkpoint(head_loss, hidden, labels, use_reentrant=False, preserve_rng_state=False)
            else:
                part, hit = head_loss(hidden, labels)
            loss = loss + part
            correct += hit.detach().sum()
            errors.scatter_add_(0, ix[:, 0], (~hit.detach()).long())
        return {"loss_sum": loss, "count": valid.sum(), "correct": correct, "row_errors": errors}

    def forward(self, input_ids, sup=None, *, targets=None, parallel=None, offset=0, compact=False):
        if targets is not None:
            return self.forward_loss(input_ids, targets, sup, parallel, offset)
        x = self.encode(input_ids)
        if compact and sup is not None:
            counts = sup.sum(1)
            if int(counts.min()) < 1 or not bool((counts == counts[0]).all()):
                raise ValueError("compact logits require equal positive query counts per row")
            x = x[sup].reshape(input_ids.shape[0], int(counts[0]), self.dim)
        logits = self.lm_head(self.final_norm(x))
        if sup is not None and not compact:
            logits = logits * sup.unsqueeze(-1)
        return logits
