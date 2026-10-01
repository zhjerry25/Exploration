"""Stack model: attention push/pop with query-written block scores.

Doctrine: TRAIN DENSE, INFER SPARSE. Text cannot be losslessly compressed,
and a block that does not know the query standard will be misjudged
(attention scatter). So the block "summary" is written FROM the query, on
demand, and never stored: every supervised position's query directly probes
every visible block's RAW keys, and the block's score is its exact attention
partition function

    s_j(t) = logsumexp_{i in block j} (q_t . k_i / sqrt(hd))

i.e. "how much attention mass would this block receive". Selection (per head,
hard top-m, no_grad) pops the winning blocks into the token-level fine read.
There is no compressor, no stored summary, no tree, no query-rewrite chain:
encode -> score -> select -> read, all parallel per position.

The score path shares q and raw_k with the fine read, so selection is the
EXACT block mass of the read semantics — a principled top-k approximation of
full attention. At topk >= visible blocks the candidate set is exactly the
causal prefix [0, t] and the model is EXACTLY a dense causal attention with a
per-block gate bias (test-guarded). That degenerate is not an approximation
but the same function evaluated densely, which is what makes the doctrine
valid here: training stays dense (forward_loss always uses the dense operator), and the
sparse top-m machinery runs only at inference, where RoPE-free scoring and
fixed-shape ops make length extrapolation structural rather than learned.
The gate is the harsh log_softmax over block scores (near-binary margins are
G-invariant; relaxed gates destroy extrapolation).

Dense read path (training + short contexts, topk >= blocks): one q.K^T
matmul serves both the block partition functions and the fine read — one
fused causal attention, no top-k, no gathers.

Sparse read path (inference, topk < blocks): exact per-head top-m selection
over block scores, gathers popped blocks + the local window, one softmax.
Autograd-capable but only ever run under no_grad by the driver.

Architecture spec (arch): a comma string of L (local halo encoder block) and
G (global read round) items. `XxN` = N independently-weighted X layers;
`(X)xN` = N passes of one weight-shared X layer (cycling adds compute passes,
not parameters). All L must precede all G (encode -> read). Examples:
"Lx2,G" (default), "Lx2,(G)x4" (read-side cycling), "Lx4,Gx2" (both sides
deeper, independent weights).

All G rounds share one raw K/V (written ONCE after the last L and never
rewritten) and one push/top-m/gate/pop machinery; only the query-side state z
evolves across rounds: z <- z + read_out(ctx(z)); z <- z + ffn(z).

KV-cache story: raw K/V is written once at token granularity and never
rewritten; the push pass only READS cached keys, the pop pass reads cached
K/V of selected blocks + the local window. Nothing else is stored.

Causality: position t (block k) reads the local window [(k-1)*b, t] directly
and may select blocks j <= k-2 (fully covered, strictly in the past); block
k-1 is covered by the local window and needs no selection.

sup: optional [B,N] bool loss mask — push/pop runs only at supervised
positions (retrieval tasks: a handful per sequence); sup=None supervises all
positions (leak tests, LM). Logits at unsupervised positions are zero.
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
from ..ops.attention import dense_attention
from ..runtime.parallel import ParallelContext


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
                 arch="Lx2,G", ffn_ratio=4, pos_chunk=0, backend="auto",
                 checkpoint_chunks=False, encoder_chunk=1024, loss_chunk=128):
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
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.block_size, self.topk, self.pos_chunk = block_size, topk, pos_chunk
        self.arch = arch
        if backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be auto, torch, or triton")
        if encoder_chunk < block_size or loss_chunk < 1:
            raise ValueError("encoder_chunk >= block_size and loss_chunk > 0 required")
        self.backend, self.checkpoint_chunks = backend, checkpoint_chunks
        self.loss_chunk = loss_chunk
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
                                              ffn_ratio, query_chunk_size=max(1, encoder_chunk // block_size),
                                              checkpoint_chunks=checkpoint_chunks)
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

    def encode_shard(self, input_ids, parallel, offset):
        x = self.embedding(input_ids)
        for layer in self.local:
            halo = parallel.halo(x, self.block_size)
            x = layer(x, previous=halo, start=offset)
        raw_k, raw_v = self.raw_kv(x, None)
        return x, parallel.to_heads(raw_k), parallel.to_heads(raw_v)

    def _dense_round(self, rd, z, k, v, positions, parallel):
        def project(hidden):
            return rd.wq(rd.q_norm(hidden))
        def finish(hidden, context):
            hidden = hidden + rd.read_out(context)
            return hidden + rd.ffn(rd.ffn_norm(hidden))
        use_ckpt = self.checkpoint_chunks and torch.is_grad_enabled()
        q = (checkpoint(project, z, use_reentrant=False, preserve_rng_state=False)
             if use_ckpt else project(z)).reshape(*z.shape[:2], self.heads, self.hd)
        q = parallel.to_heads(q)
        ctx = dense_attention(q, k, v, positions, self.block_size, self.backend)
        ctx = parallel.to_sequence(ctx).reshape(*z.shape[:2], self.dim)
        return (checkpoint(finish, z, ctx, use_reentrant=False, preserve_rng_state=False)
                if use_ckpt else finish(z, ctx))

    def forward_loss(self, input_ids, targets, sup=None, parallel=None, offset=0):
        """Always-dense training; return SUM loss and counts, never [B,N,V].

        Inputs are local sequence shards for CP. Every CP peer participates,
        even when it owns no supervised tokens. The trainer normalizes by
        the actual global token count before DDP's gradient averaging.
        """
        parallel = parallel or ParallelContext()
        if input_ids.ndim != 2 or input_ids.shape != targets.shape or min(input_ids.shape) < 1:
            raise ValueError("ids and targets must have identical nonempty [B,N] shape")
        if self.heads % parallel.cp_size:
            raise ValueError("heads must be divisible by context_parallel")
        if sup is None:
            sup = targets != -100
        if sup.shape != targets.shape or sup.dtype != torch.bool:
            raise ValueError("sup must be bool [B,N]")
        sup = sup & (targets != -100)
        batch, n = input_ids.shape
        counts = sup.sum(1)
        width = parallel.max_query_count(int(counts.max()), input_ids.device)
        # Sort only integer positions, not the large feature tensor.
        indices = torch.arange(n, device=input_ids.device)[None].expand(batch, -1)
        indices = torch.where(sup, indices, n).sort(1).values
        if width > n:
            indices = F.pad(indices, (0, width-n), value=n)
        indices = indices[:, :width]
        valid = indices < n
        safe = indices.clamp_max(n-1)
        pos = torch.where(valid, indices + offset, -1)
        selected_targets = targets.gather(1, safe).masked_fill(~valid, -100)
        x, k, v = self.encode_shard(input_ids, parallel, offset)
        rows = torch.arange(batch, device=x.device)[:, None]
        losses, hits = [], []
        step = self.pos_chunk or 128
        for lo in range(0, width, step):
            hi = min(lo+step, width)
            z = x[rows, safe[:, lo:hi]]
            positions = parallel.gather_positions(pos[:, lo:hi])
            for rd in self.reads:
                # Do NOT checkpoint collectives: all ranks must execute the
                # same communication order even with unsupervised shards.
                z = self._dense_round(rd, z, k, v, positions, parallel)
            for j in range(0, hi-lo, self.loss_chunk):
                target = selected_targets[:, lo+j:min(lo+j+self.loss_chunk, hi)]
                def head_loss(hidden, labels):
                    logits = self.lm_head(self.final_norm(hidden)).float()
                    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                           ignore_index=-100, reduction="sum")
                    hit = (logits.argmax(-1) == labels) | (labels == -100)
                    return loss, hit
                h = z[:, j:j+self.loss_chunk]
                if self.checkpoint_chunks and torch.is_grad_enabled():
                    loss, hit = checkpoint(head_loss, h, target, use_reentrant=False,
                                           preserve_rng_state=False)
                else:
                    loss, hit = head_loss(h, target)
                losses.append(loss)
                hits.append(hit.detach())
        hit = torch.cat(hits, 1)
        return {"loss_sum": torch.stack(losses).sum(), "count": valid.sum(),
                "correct": (hit & valid).sum(), "row_errors": ((~hit) & valid).sum(1),
                "positions": pos, "hits": hit}

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
        elems/pos) to ~2^29 elements, so eval-time peaks stay modest even at
        bs16 (<=~11GB bf16)."""
        if self.pos_chunk > 0:
            return self.pos_chunk
        groups = (n + self.block_size - 1) // self.block_size
        m = min(self.topk, groups)
        elems_per_pos = self.heads * (n + 2 * m * self.block_size * self.hd)
        return min(512, max(16, (1 << 29) // max(elems_per_pos, 1)))

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

    def _prof_tick(self, device):
        """Phase timing hook (set self._prof to enable); syncs device first."""
        if not getattr(self, "_prof", False):
            return None
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter()

    def forward(self, input_ids, sup=None, *, targets=None, parallel=None, offset=0,
                compact=False):
        if targets is not None:
            return self.forward_loss(input_ids, targets, sup, parallel, offset)
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        batch, n = input_ids.shape
        b = self.block_size
        groups = (n + b - 1) // b
        t0 = self._prof_tick(input_ids.device)
        x, raw_k, raw_v = self.encode(input_ids)
        t1 = self._prof_tick(input_ids.device)
        pos, s_n = self.supervised_positions(sup, n)
        if pos is None:
            pos = torch.arange(n, device=x.device)[None].expand(batch, n)
        offs = torch.arange(2 * b, device=x.device)
        dense = self.topk >= groups
        env = None
        if not dense and torch.is_grad_enabled():
            # Differentiable sparse reference is retained for mathematical
            # checks. Production loss always dispatches forward_loss above.
            pad = groups*b-n
            kb = F.pad(raw_k, (0, 0, 0, 0, 0, pad)).reshape(batch, groups, b, self.heads, self.hd)
            cover_end = (torch.arange(groups, device=x.device)+1)*b
            env = dict(n=n, b=b, groups=groups, kb=kb, raw_k=raw_k, raw_v=raw_v,
                       rows3=torch.arange(batch, device=x.device)[:, None, None])
        z_rows = torch.arange(batch, device=x.device)[:, None]
        outs = []
        chunk = self._chunk_size(n)
        for start in range(0, s_n, chunk):
            sl = slice(start, min(s_n, start + chunk))
            z = x[z_rows, pos[:, sl]]
            pc = pos[:, sl]
            if not dense and torch.is_grad_enabled():
                block_of = pc // b
                valid_from = ((block_of-1) * b).clamp_min(0)
                src = pc[:, :, None] - (2*b-1) + offs
                env.update(pos=pc, src=src,
                           src_valid=(src >= valid_from[:, :, None]) & (src >= 0),
                           visible=cover_end[None, None, :] <= ((block_of-1)*b)[:, :, None])
            for rd in self.reads:
                if dense:
                    z = self._dense_round(rd, z, raw_k, raw_v, pc, ParallelContext())
                elif not torch.is_grad_enabled():
                    from ..ops.sparse import sparse_attention, TensorPages
                    q = rd.wq(rd.q_norm(z)).reshape(batch, z.shape[1], self.heads, self.hd)
                    ctx = sparse_attention(q, TensorPages(raw_k, raw_v,
                                           page_tokens=max(b, 65536//b*b)),
                                           pc, b, min(self.topk, groups), self.backend)
                    z = z + rd.read_out(ctx.reshape(batch, z.shape[1], self.dim))
                    z = z + rd.ffn(rd.ffn_norm(z))
                else:
                    z = self._read_round_sparse(rd, z, slice(None), env)
            outs.append(z)
        t2 = self._prof_tick(input_ids.device)
        z = torch.cat(outs, dim=1)
        sup_logits = self.lm_head(self.final_norm(z))
        if compact:
            return sup_logits
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
        rd = self.reads[0]
        z_rows = torch.arange(batch, device=x.device)[:, None]
        keep = min(self.topk, groups)
        cov_sum = cov_cnt = mar_sum = mar_cnt = 0.0
        hit_sum = hit_cnt = 0.0
        for start in range(0, s_n, self._chunk_size(n)):
            stop = min(s_n, start + self._chunk_size(n))
            cnt = stop - start
            vis = cover_end[None, None, :] <= ((block_of[:, start:stop]-1)*b)[:, :, None]
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
