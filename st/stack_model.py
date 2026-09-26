"""Stack model: attention push/pop with query-written block scores.

Doctrine: text cannot be losslessly compressed, and a block that does not know
the query standard will be misjudged (attention scatter). So the block
"summary" is written FROM the query, on demand, and never stored: every
supervised position's query directly probes every visible block's RAW keys,
and the block's score is its exact attention partition function

    s_j(t) = logsumexp_{i in block j} (q_t . k_i / sqrt(hd))

i.e. "how much attention mass would this block receive". Selection (per head,
hard top-m, no_grad) pops the winning blocks into the token-level fine read.
There is no compressor, no stored summary, no tree, no query-rewrite chain:
encode -> score -> select -> read, all parallel per position.

The score path shares q and raw_k with the fine read, so selection is the
EXACT block mass of the read semantics — a principled top-k approximation of
full attention (at topk >= visible blocks it is exactly dense, which makes
small-n ignition behave like a plain transformer). The gate is the harsh
log_softmax over block scores (near-binary margins are G-invariant; relaxed
gates destroy extrapolation).

KV-cache story: raw K/V is written once at token granularity and never
rewritten; the push pass only READS cached keys, the pop pass reads cached
K/V of selected blocks + the local window. Nothing else is stored.

Causality: position t (block k) reads the local window [(k-1)*b, t] directly
and may select blocks j <= k-2 (fully covered, strictly in the past); block
k-1 is covered by the local window and needs no selection.

sup: optional [B,N] bool loss mask — push/pop runs only at supervised
positions (retrieval tasks: a handful per sequence); sup=None supervises all
positions (leak tests, LM). Logits at unsupervised positions are zero.
RoPE lives only in the local encoder; the score/read paths are NoPE so
retrieval is distance-independent and length extrapolation is structural.
"""
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (HaloMemoryBlock, RotaryEmbedding, FeedForward,
                     KVProjection, _gather_heads)


class StackModel(nn.Module):
    """forward([B,N], sup=None) -> [B,N,vocab_size]."""

    def __init__(self, vocab_size, dim=256, heads=4, block_size=16, topk=64,
                 local_layers=2, ffn_ratio=4, pos_chunk=256):
        super().__init__()
        integers = {"vocab_size": vocab_size, "dim": dim, "heads": heads,
                    "block_size": block_size, "topk": topk,
                    "local_layers": local_layers, "pos_chunk": pos_chunk}
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if dim % heads or (dim // heads) % 2:
            raise ValueError("dim / heads must be an even integer for RoPE")
        if not math.isfinite(ffn_ratio) or ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be finite and positive")
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.block_size, self.topk, self.pos_chunk = block_size, topk, pos_chunk
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rope = RotaryEmbedding(self.hd)
        self.local = nn.ModuleList(
            HaloMemoryBlock(dim, heads, block_size, self.rope, ffn_ratio,
                            query_chunk_size=64)
            for _ in range(local_layers))
        self.raw_kv = KVProjection(dim, heads, self.rope, use_rope=False)
        self.q_norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.read_out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def encode(self, input_ids):
        """Tokens -> (x, raw_k, raw_v). Write-once content, linear cost."""
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        n = input_ids.shape[1]
        x = self.embedding(input_ids)
        positions = torch.arange(n, device=x.device)
        for block in self.local:
            x = block(x, positions)
        raw_k, raw_v = self.raw_kv(x, positions)
        return x, raw_k, raw_v

    @staticmethod
    def supervised_positions(sup, n):
        """-> [B,S] position ids. sup=None supervises all positions."""
        if sup is None:
            return None, n
        counts = sup.sum(1)
        s = int(counts[0])
        if s < 1 or not bool((counts == s).all()):
            raise ValueError("sup must mark the same positive count per row")
        nz = sup.nonzero()  # row-major: sorted by row, then position
        return nz[:, 1].reshape(sup.shape[0], s), s

    def push_scores(self, q, kb, visible):
        """PUSH: exact per-block attention partition function.

        q [B,c,H,hd]; kb [B,G,b,H,hd]; visible [B,c,G] ->
        block scores [B,c,H,G] (masked -inf), token scores [B,c,H,G,b]."""
        ts = torch.einsum('bchd,bgwhd->bchgw', q, kb) / math.sqrt(self.hd)
        s = ts.logsumexp(-1)
        s = s.masked_fill(~visible[:, :, None, :], float('-inf'))
        return s

    def forward(self, input_ids, sup=None):
        batch, n = input_ids.shape
        b = self.block_size
        groups = (n + b - 1) // b
        x, raw_k, raw_v = self.encode(input_ids)
        pos, s_n = self.supervised_positions(sup, n)
        if pos is None:
            pos = torch.arange(n, device=x.device)[None].expand(batch, n)
        block_of = pos // b
        valid_from = torch.where(block_of >= 1, (block_of - 1) * b,
                                 torch.zeros_like(block_of))
        offs = torch.arange(2 * b, device=x.device)
        src = pos[:, :, None] - (2 * b - 1) + offs           # local window
        src_valid = (src >= valid_from[:, :, None]) & (src >= 0)
        # block-view of the raw keys for the push pass (padded tail never
        # satisfies visibility, so padding garbage is always masked)
        pad = groups * b - n
        kb = F.pad(raw_k, (0, 0, 0, 0, 0, pad)).reshape(batch, groups, b,
                                                        self.heads, self.hd)
        cover_end = (torch.arange(groups, device=x.device) + 1) * b
        indexed_end = (block_of - 1) * b                     # blocks j <= k-2
        visible = cover_end[None, None, :] <= indexed_end[:, :, None]
        rows = torch.arange(batch, device=x.device)[:, None, None]
        k_loc = raw_k[rows, src.clamp(0, n - 1)]             # [B,S,2b,H,hd]
        v_loc = raw_v[rows, src.clamp(0, n - 1)]
        z_rows = torch.arange(batch, device=x.device)[:, None]
        outs = []
        for start in range(0, s_n, self.pos_chunk):
            stop = min(s_n, start + self.pos_chunk)
            cnt = stop - start
            pos_c = pos[:, start:stop]
            x_c = x[z_rows, pos_c]
            q = self.wq(self.q_norm(x_c)).reshape(batch, cnt, self.heads,
                                                  self.hd)
            s = self.push_scores(q, kb, visible[:, start:stop])  # [B,c,H,G]
            with torch.no_grad():
                keep = min(self.topk, groups)
                top, sel = s.topk(keep, dim=-1)
                sel_ok = torch.isfinite(top)
            gate = torch.nan_to_num(s - s.logsumexp(-1, keepdim=True),
                                    nan=0.0, neginf=0.0)
            gate_sel = torch.gather(gate, 3, sel)
            gate_sel = torch.where(sel_ok, gate_sel,
                                   torch.zeros((), dtype=gate.dtype,
                                               device=gate.device))
            # POP: expand selected blocks into raw tokens, per head
            tok = sel[..., None] * b + torch.arange(b, device=x.device)
            tok = tok.reshape(batch, cnt, self.heads, -1)         # [B,c,H,m*b]
            tok_ok = (sel_ok[..., None].expand(-1, -1, -1, -1, b)
                      .reshape(batch, cnt, self.heads, -1)
                      & (tok < n) & (tok <= pos_c[:, :, None, None]))
            k_rem = _gather_heads(raw_k.transpose(1, 2),
                                  tok.clamp(0, n - 1).permute(0, 2, 1, 3))
            v_rem = _gather_heads(raw_v.transpose(1, 2),
                                  tok.clamp(0, n - 1).permute(0, 2, 1, 3))
            k_rem = k_rem.permute(0, 2, 1, 3, 4)                  # [B,c,H,m*b,hd]
            v_rem = v_rem.permute(0, 2, 1, 3, 4)
            bias = gate_sel[..., None].expand(-1, -1, -1, -1, b)
            bias = bias.reshape(batch, cnt, self.heads, -1)
            s_loc = torch.einsum('bchd,bcwhd->bchw', q,
                                 k_loc[:, start:stop]) / math.sqrt(self.hd)
            s_rem = torch.einsum('bchd,bchkd->bchk', q, k_rem) \
                / math.sqrt(self.hd) + bias
            all_scores = torch.cat([s_loc, s_rem], dim=-1)
            mask = torch.cat([src_valid[:, start:stop, None]
                              .expand(-1, -1, self.heads, -1), tok_ok], dim=-1)
            probs = all_scores.masked_fill(~mask, float('-inf')).softmax(-1)
            width = s_loc.shape[-1]
            ctx = torch.einsum('bchw,bcwhd->bchd', probs[..., :width],
                               v_loc[:, start:stop]) \
                + torch.einsum('bchk,bchkd->bchd', probs[..., width:], v_rem)
            ctx = ctx.reshape(batch, cnt, self.dim)
            z = x_c + self.read_out(ctx)
            outs.append(z + self.ffn(self.ffn_norm(z)))
        z = torch.cat(outs, dim=1)
        sup_logits = self.lm_head(self.final_norm(z))
        logits = sup_logits.new_zeros(batch, n, sup_logits.shape[-1])
        logits[z_rows, pos] = sup_logits
        return logits
