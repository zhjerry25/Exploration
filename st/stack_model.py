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

Architecture spec (arch): a comma string of L (local halo encoder block) and
G (global read round) items. `XxN` = N independently-weighted X layers;
`(X)xN` = N passes of one weight-shared X layer (cycling adds compute passes,
not parameters). All L must precede all G (encode -> read). Examples:
"Lx2,G" (default, == the original 2-local + 1-read model), "Lx2,(G)x4"
(read-side cycling), "Lx4,Gx2" (both sides deeper, independent weights).

All G rounds share one raw K/V (written ONCE after the last L and never
rewritten) and one push/top-m/gate/pop machinery; only the query-side state z
evolves across rounds: z <- z + read_out(ctx(z)); z <- z + ffn(z).

KV-cache story: raw K/V is written once at token granularity and never
rewritten; the push pass only READS cached keys, the pop pass reads cached
K/V of selected blocks + the local window. Nothing else is stored.

Causality: position t (block k) reads the local window [(k-1)*b, t] directly
and may select blocks j <= k-2 (fully covered, strictly in the past); block
k-1 is covered by the local window and needs no selection.

Dense fast path: when topk >= total blocks, every visible block is selected,
the candidate set is exactly the causal prefix [0, t], and push+pop collapse
into a single causal attention with a per-block gate bias (one score matmul
serves both the block partition functions and the fine read). Numerically
equivalent to the sparse path (guarded by tests), much cheaper at small n.

Memory: the sparse read materialises ~H*(n + 2*m*b*hd) temporaries per
supervised position (push scores + per-head block gathers), which autograd
would retain across ALL chunks until the single backward — ~90GB at
n=4096/bs16. During training each read round therefore runs under
torch.utils.checkpoint (grad_ckpt=True, default): only the round input z is
kept, chunk temporaries are recomputed during backward. Peak drops to one
chunk's temporaries (~5GB at bs16) at the cost of one extra read-side
forward. Disable (--grad_ckpt 0) on >=80GB cards for full speed. The read
path is RNG-free, so recomputation is exact.

sup: optional [B,N] bool loss mask — push/pop runs only at supervised
positions (retrieval tasks: a handful per sequence); sup=None supervises all
positions (leak tests, LM). Logits at unsupervised positions are zero.
RoPE lives only in the local encoder; the score/read paths are NoPE so
retrieval is distance-independent and length extrapolation is structural.
"""
import math
import time
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import (HaloMemoryBlock, RotaryEmbedding, FeedForward,
                     KVProjection, _gather_heads)


def parse_arch(arch):
    """'Lx2,(G)x4' -> [(kind, count, shared), ...] in application order.

    L = local halo encoder block; G = global read round. `XxN` repeats X with
    independent weights, `(X)xN` cycles one shared-weight X. All L must
    precede all G (encode -> read); bare `X` means `Xx1`.
    """
    if not isinstance(arch, str) or not arch.strip():
        raise ValueError("arch must be a nonempty spec like 'Lx2,G'")
    spec = []
    for item in arch.split(","):
        item = item.strip()
        shared = item.startswith("(")
        if shared:
            kind, sep, count = item[1:].partition(")x")
            if not sep:
                raise ValueError(f"shared layers must look like '(X)xN': {item!r}")
        else:
            kind, _, count = item.partition("x")
            count = count or "1"
        if kind not in ("L", "G"):
            raise ValueError(f"unknown layer kind in arch item {item!r} (want L or G)")
        if not count.isdigit() or int(count) < 1:
            raise ValueError(f"repeat count must be a positive integer in {item!r}")
        spec.append((kind, int(count), shared))
    kinds = [k for k, _, _ in spec]
    if "G" in kinds and "L" in kinds:
        first_g = kinds.index("G")
        last_l = len(kinds) - 1 - kinds[::-1].index("L")
        if first_g < last_l:
            raise ValueError("all L layers must precede all G layers (encode -> read)")
    return spec


class GlobalReadBlock(nn.Module):
    """One global read round's parameters (query norm/proj, readout, FFN).

    The push/top-m/gate/pop plumbing lives in StackModel because raw K/V are
    shared across all rounds and never rewritten; this block only owns the
    query-side weights that turn the current z into a query and absorb the
    retrieved context.
    """

    def __init__(self, dim, ffn_ratio):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.read_out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)


class StackModel(nn.Module):
    """forward([B,N], sup=None) -> [B,N,vocab_size]."""

    def __init__(self, vocab_size, dim=256, heads=4, block_size=16, topk=64,
                 arch=None, local_layers=None, ffn_ratio=4, pos_chunk=0,
                 grad_ckpt=True):
        super().__init__()
        integers = {"vocab_size": vocab_size, "dim": dim, "heads": heads,
                    "block_size": block_size, "topk": topk}
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(pos_chunk, bool) or not isinstance(pos_chunk, Integral) \
                or pos_chunk < 0:
            raise ValueError("pos_chunk must be a non-negative integer (0 = auto)")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if dim % heads or (dim // heads) % 2:
            raise ValueError("dim / heads must be an even integer for RoPE")
        if not math.isfinite(ffn_ratio) or ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be finite and positive")
        if arch is None:
            arch = f"Lx{local_layers if local_layers is not None else 2},G"
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.block_size, self.topk, self.pos_chunk = block_size, topk, pos_chunk
        self.grad_ckpt = bool(grad_ckpt)
        self.arch = arch
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rope = RotaryEmbedding(self.hd)
        shared = {}
        local, reads = [], []
        for kind, count, is_shared in parse_arch(arch):
            for _ in range(count):
                key = kind if is_shared else None
                if key is not None and key in shared:
                    mod = shared[key]
                else:
                    if kind == "L":
                        mod = HaloMemoryBlock(dim, heads, block_size, self.rope,
                                              ffn_ratio, query_chunk_size=64)
                    else:
                        mod = GlobalReadBlock(dim, ffn_ratio)
                    if key is not None:
                        shared[key] = mod
                (local if kind == "L" else reads).append(mod)
        self.local = nn.ModuleList(local)
        self.reads = nn.ModuleList(reads)
        self.raw_kv = KVProjection(dim, heads, self.rope, use_rope=False)
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
        block scores [B,c,H,G] (masked -inf)."""
        ts = torch.einsum('bchd,bgwhd->bchgw', q, kb) / math.sqrt(self.hd)
        s = ts.logsumexp(-1)
        s = s.masked_fill(~visible[:, :, None, :], float('-inf'))
        return s

    def _chunk_size(self, n):
        """Supervised positions per chunk; auto mode bounds the biggest
        temporary (push scores ~ H*n elems/pos, gathers ~ 2*H*m*b*hd
        elems/pos) to ~2^28 elements (~0.5GB bf16)."""
        if self.pos_chunk > 0:
            return self.pos_chunk
        groups = (n + self.block_size - 1) // self.block_size
        m = min(self.topk, groups)
        elems_per_pos = self.heads * (n + 2 * m * self.block_size * self.hd)
        return min(256, max(16, (1 << 28) // max(elems_per_pos, 1)))

    def _read_round_sparse(self, rd, z, sl, env):
        """One G round, exact top-m plumbing (general case). The local-window
        gathers are sliced per chunk here (not hoisted) so checkpoint
        recomputation owns them."""
        batch, cnt, _ = z.shape
        n, b, groups = env["n"], env["b"], env["groups"]
        pos_c = env["pos"][:, sl]
        src_c = env["src"][:, sl]                                # [B,c,2b]
        rows = env["rows3"]
        k_loc = env["raw_k"][rows, src_c.clamp(0, n - 1)]        # [B,c,2b,H,hd]
        v_loc = env["raw_v"][rows, src_c.clamp(0, n - 1)]
        q = rd.wq(rd.q_norm(z)).reshape(batch, cnt, self.heads, self.hd)
        s = self.push_scores(q, env["kb"], env["visible"][:, sl])  # [B,c,H,G]
        with torch.no_grad():
            keep = min(self.topk, groups)
            top, sel = s.topk(keep, dim=-1)
            sel_ok = torch.isfinite(top)
        gate = torch.nan_to_num(s - s.logsumexp(-1, keepdim=True),
                                nan=0.0, neginf=0.0)
        gate_sel = torch.gather(gate, 3, sel)
        gate_sel = torch.where(sel_ok, gate_sel,
                               torch.zeros((), dtype=gate.dtype, device=gate.device))
        # POP: expand selected blocks into raw tokens, per head
        tok = sel[..., None] * b + torch.arange(b, device=z.device)
        tok = tok.reshape(batch, cnt, self.heads, -1)              # [B,c,H,m*b]
        tok_ok = (sel_ok[..., None].expand(-1, -1, -1, -1, b)
                  .reshape(batch, cnt, self.heads, -1)
                  & (tok < n) & (tok <= pos_c[:, :, None, None]))
        k_rem = _gather_heads(env["raw_k"].transpose(1, 2),
                              tok.clamp(0, n - 1).permute(0, 2, 1, 3))
        v_rem = _gather_heads(env["raw_v"].transpose(1, 2),
                              tok.clamp(0, n - 1).permute(0, 2, 1, 3))
        k_rem = k_rem.permute(0, 2, 1, 3, 4)                       # [B,c,H,m*b,hd]
        v_rem = v_rem.permute(0, 2, 1, 3, 4)
        bias = gate_sel[..., None].expand(-1, -1, -1, -1, b)
        bias = bias.reshape(batch, cnt, self.heads, -1)
        s_loc = torch.einsum('bchd,bcwhd->bchw', q,
                             k_loc) / math.sqrt(self.hd)
        s_rem = torch.einsum('bchd,bchkd->bchk', q, k_rem) \
            / math.sqrt(self.hd) + bias
        all_scores = torch.cat([s_loc, s_rem], dim=-1)
        mask = torch.cat([env["src_valid"][:, sl, None]
                          .expand(-1, -1, self.heads, -1), tok_ok], dim=-1)
        probs = all_scores.masked_fill(~mask, float('-inf')).softmax(-1)
        width = s_loc.shape[-1]
        ctx = torch.einsum('bchw,bcwhd->bchd', probs[..., :width],
                           v_loc) \
            + torch.einsum('bchk,bchkd->bchd', probs[..., width:], v_rem)
        ctx = ctx.reshape(batch, cnt, self.dim)
        z = z + rd.read_out(ctx)
        return z + rd.ffn(rd.ffn_norm(z))

    def _read_round_dense(self, rd, z, sl, env):
        """One G round when topk >= groups: the candidate set is exactly the
        causal prefix [0, t], so push and pop collapse into a single causal
        attention with a per-block gate bias. One q.K^T serves both the block
        partition functions and the fine read."""
        batch, cnt, _ = z.shape
        n, b, groups, pad = env["n"], env["b"], env["groups"], env["pad"]
        pos_c = env["pos"][:, sl]
        q = rd.wq(rd.q_norm(z)).reshape(batch, cnt, self.heads, self.hd)
        qh = q.permute(0, 2, 1, 3).reshape(batch * self.heads, cnt, self.hd)
        kh = env["raw_k"].permute(0, 2, 3, 1).reshape(batch * self.heads,
                                                      self.hd, n)
        ts = torch.bmm(qh, kh).view(batch, self.heads, cnt, n) \
            .permute(0, 2, 1, 3) / math.sqrt(self.hd)             # [B,c,H,n]
        tsp = F.pad(ts, (0, pad)).view(batch, cnt, self.heads, groups, b)
        s = tsp.logsumexp(-1)                                     # [B,c,H,G]
        s = s.masked_fill(~env["visible"][:, sl, None, :], float('-inf'))
        gate = torch.nan_to_num(s - s.logsumexp(-1, keepdim=True),
                                nan=0.0, neginf=0.0)
        bias = gate[:, :, :, env["block_ids"]]                    # [B,c,H,n]
        causal = pos_c[:, :, None, None] >= env["token_ids"][None, None, None, :]
        probs = (ts + bias).masked_fill(~causal, float('-inf')).softmax(-1)
        vh = env["raw_v"].permute(0, 2, 1, 3)                     # [B,H,n,hd]
        ctx = torch.bmm(probs.permute(0, 2, 1, 3).reshape(batch * self.heads,
                                                         cnt, n),
                        vh.reshape(batch * self.heads, n, self.hd))
        ctx = ctx.view(batch, self.heads, cnt, self.hd) \
            .permute(0, 2, 1, 3).reshape(batch, cnt, self.dim)
        z = z + rd.read_out(ctx)
        return z + rd.ffn(rd.ffn_norm(z))

    def _prof_tick(self, device):
        """Phase timing hook (set self._prof to enable); syncs device first."""
        if not getattr(self, "_prof", False):
            return None
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter()

    def forward(self, input_ids, sup=None):
        batch, n = input_ids.shape
        b = self.block_size
        groups = (n + b - 1) // b
        t0 = self._prof_tick(input_ids.device)
        x, raw_k, raw_v = self.encode(input_ids)
        t1 = self._prof_tick(input_ids.device)
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
        env = dict(n=n, b=b, groups=groups, pad=pad, pos=pos, kb=kb,
                   visible=visible, raw_k=raw_k, raw_v=raw_v,
                   src=src, src_valid=src_valid,
                   rows3=torch.arange(batch, device=x.device)[:, None, None],
                   block_ids=torch.arange(n, device=x.device) // b,
                   token_ids=torch.arange(n, device=x.device))
        dense = self.topk >= groups
        round_fn = self._read_round_dense if dense else self._read_round_sparse
        z_rows = torch.arange(batch, device=x.device)[:, None]
        outs = []
        chunk = self._chunk_size(n)
        for start in range(0, s_n, chunk):
            sl = slice(start, min(s_n, start + chunk))
            z = x[z_rows, pos[:, sl]]
            for rd in self.reads:
                if self.grad_ckpt and self.training and torch.is_grad_enabled():
                    # keep only z across chunks; recompute the big temporaries
                    # (push scores, per-head gathers) during backward
                    z = checkpoint(round_fn, rd, z, sl, env,
                                   use_reentrant=False)
                else:
                    z = round_fn(rd, z, sl, env)
            outs.append(z)
        t2 = self._prof_tick(input_ids.device)
        z = torch.cat(outs, dim=1)
        sup_logits = self.lm_head(self.final_norm(z))
        logits = sup_logits.new_zeros(batch, n, sup_logits.shape[-1])
        logits[z_rows, pos] = sup_logits
        t3 = self._prof_tick(input_ids.device)
        if t0 is not None:
            self._prof_stats = {"enc_s": round(t1 - t0, 3),
                                "read_s": round(t2 - t1, 3),
                                "head_s": round(t3 - t2, 3)}
        return logits

    @torch.no_grad()
    def diagnose(self, idx, sup=None, needle_pos=None):
        """Encode + push only (no pop): selection quality of the first G round.

        Returns sel_cov (attention mass covered by the top-m selected blocks,
        i.e. 1 - uncovered tail eps), gate_margin (lowest selected block score
        minus highest unselected visible score), and, when needle_pos [B] is
        given (passkey), needle_block_hit (fraction of heads whose top-m
        contains a needle block at the first supervised position, over rows
        where the needle is outside the local window)."""
        if not self.reads:
            return {}
        self.eval()
        batch, n = idx.shape
        b = self.block_size
        groups = (n + b - 1) // b
        x, raw_k, _ = self.encode(idx)
        pos, s_n = self.supervised_positions(sup, n)
        if pos is None:
            pos = torch.arange(n, device=x.device)[None].expand(batch, n)
        block_of = pos // b
        pad = groups * b - n
        kb = F.pad(raw_k, (0, 0, 0, 0, 0, pad)).reshape(batch, groups, b,
                                                        self.heads, self.hd)
        cover_end = (torch.arange(groups, device=x.device) + 1) * b
        visible = cover_end[None, None, :] <= ((block_of - 1) * b)[:, :, None]
        rd = self.reads[0]
        z_rows = torch.arange(batch, device=x.device)[:, None]
        keep = min(self.topk, groups)
        cov_sum = cov_cnt = mar_sum = mar_cnt = 0.0
        hit_sum = hit_cnt = 0.0
        for start in range(0, s_n, self._chunk_size(n)):
            stop = min(s_n, start + self._chunk_size(n))
            cnt = stop - start
            vis = visible[:, start:stop]                        # [B,c,G]
            z = x[z_rows, pos[:, start:stop]]
            q = rd.wq(rd.q_norm(z)).reshape(batch, cnt, self.heads, self.hd)
            s = self.push_scores(q, kb, vis)                    # [B,c,H,G]
            top, sel = s.topk(keep, dim=-1)
            sel_ok = torch.isfinite(top)
            p = s.softmax(-1)                                   # nan if no visible
            p_sel = torch.where(sel_ok, p.gather(3, sel),
                                torch.zeros((), dtype=p.dtype, device=p.device))
            cov = p_sel.sum(-1)                                 # [B,c,H]
            has_vis = vis.any(-1, keepdim=True)                 # [B,c,1]
            cov_sum += cov[has_vis.expand_as(cov)].sum().item()
            cov_cnt += has_vis.sum().item() * self.heads
            sel_mask = torch.zeros_like(s, dtype=torch.bool).scatter(3, sel, sel_ok)
            unsel_vis = vis[:, :, None, :] & ~sel_mask
            s_sel_min = s.masked_fill(~sel_mask, float('inf')).min(-1).values
            s_unsel_max = s.masked_fill(~unsel_vis, float('-inf')).max(-1).values
            mvalid = sel_mask.any(-1) & unsel_vis.any(-1)       # [B,c,H]
            mar_sum += (s_sel_min - s_unsel_max)[mvalid].sum().item()
            mar_cnt += mvalid.sum().item()
            if needle_pos is not None and start == 0:
                nb = (needle_pos.to(x.device)[:, None]
                      + torch.arange(6, device=x.device)) // b  # [B,6]
                k0 = int(pos[0, 0]) // b
                past = nb.max(-1).values <= k0 - 2              # outside local window
                if past.any():
                    at0 = sel_mask[:, 0]                        # [B,H,G]
                    hits = at0.gather(2, nb[:, None, :]
                                      .expand(-1, self.heads, -1)).any(-1)
                    hit_sum += hits[past].float().mean(-1).sum().item()
                    hit_cnt += int(past.sum().item())
        rec = {}
        if cov_cnt:
            rec["sel_cov"] = round(cov_sum / cov_cnt, 4)
        if mar_cnt:
            rec["gate_margin"] = round(mar_sum / mar_cnt, 4)
        if hit_cnt:
            rec["needle_block_hit"] = round(hit_sum / hit_cnt, 4)
        return rec
