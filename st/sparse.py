"""Bounded exact top-k selection and sparse read across KV pages/shards.

Scoring is still O(number of queries * context * head_dim): exact query-
written block scores cannot be computed from a compressed index. Only the
read is sparse. Both score workspace and selected-value workspace are bounded.
"""
import math

import torch
import torch.distributed as dist

from .attention import triton_supported, _work_dtype


def _score_torch(q, k, positions, b, offset):
    batch, queries, heads, dim = q.shape
    n = k.shape[1]
    pad = (-n) % b
    dtype = _work_dtype(q)
    kh = k.transpose(1, 2).to(dtype)
    if pad:
        kh = torch.nn.functional.pad(kh, (0, 0, 0, pad))
    a = q.transpose(1, 2).to(dtype) @ kh.transpose(-1, -2) / math.sqrt(dim)
    a[..., n:] = -torch.inf
    scores = a.reshape(batch, heads, queries, -1, b).logsumexp(-1).transpose(1, 2)
    blocks = torch.arange(scores.shape[-1], device=q.device) + offset//b
    visible = ((blocks[None, None, None, :] < positions[:, :, None, None]//b-1)
               & ((blocks[None, None, None, :]+1)*b <= offset+n)
               & (positions[:, :, None, None] >= 0))
    return scores.masked_fill(~visible, -torch.inf)


def _pop_torch(q, k, v, pos, ids, scores, log_zr, b, offset):
    """Oracle/fallback: one selected block at a time, no large KV gather."""
    batch, queries, heads, dim = q.shape
    dtype = _work_dtype(q)
    q = q.to(dtype)
    m = torch.full(q.shape[:3], -torch.inf, dtype=dtype, device=q.device)
    den, num = torch.zeros_like(m), torch.zeros_like(q)
    rows = torch.arange(batch, device=q.device)[:, None, None, None]
    hs = torch.arange(heads, device=q.device)[None, None, :, None]
    for j in range(-2, ids.shape[-1]):
        if j < 0:
            block = (pos//b + (j+1))[:, :, None].expand(-1, -1, heads)
            bias = torch.zeros_like(m)
            ok = pos[:, :, None] >= 0
        else:
            block = ids[..., j]
            bias = scores[..., j] - torch.where(torch.isfinite(log_zr), log_zr, 0.)
            ok = (block >= 0) & torch.isfinite(scores[..., j])
        token = block[..., None]*b + torch.arange(b, device=q.device)
        valid = ok[..., None] & (token >= offset) & (token < offset+k.shape[1]) & (token >= 0) & (token <= pos[:, :, None, None])
        local_token = (token-offset).clamp(0, k.shape[1]-1)
        kr = k[rows, local_token, hs].to(dtype)
        vr = v[rows, local_token, hs].to(dtype)
        a = (q[..., None, :]*kr).sum(-1)/math.sqrt(dim) + bias[..., None]
        a = a.masked_fill(~valid, -torch.inf)
        m1 = torch.maximum(m, a.amax(-1))
        safe = torch.where(torch.isfinite(m1), m1, 0.)
        alpha = torch.exp(m-safe)
        p = torch.exp(a-safe[..., None])
        den = den*alpha + p.sum(-1)
        num = num*alpha[..., None] + (p[..., None]*vr).sum(-2)
        m = m1
    return num/den.clamp_min(torch.finfo(dtype).tiny)[..., None], m+den.log()


def _merge_read(out, lse, part, part_lse):
    total = torch.logaddexp(lse, part_lse)
    safe = torch.where(torch.isfinite(total), total, 0.)
    merged = out*torch.exp(lse-safe)[..., None] + part*torch.exp(part_lse-safe)[..., None]
    return merged, total


class TensorPages:
    """In-core KV adapter, same protocol as the out-of-core cache."""
    def __init__(self, k, v, page_tokens=65536, offset=0):
        self.k, self.v, self.page_tokens, self.offset = k, v, page_tokens, offset
        self.length = k.shape[1]

    def pages(self, device, indices=None, keys_only=False):
        if indices is None:
            indices = range((self.length+self.page_tokens-1)//self.page_tokens)
        for page in indices:
            lo = page*self.page_tokens
            hi = min(lo+self.page_tokens, self.length)
            if hi > lo:
                key = self.k[:, lo:hi].to(device).contiguous()
                value = None if keys_only else self.v[:, lo:hi].to(device).contiguous()
                yield self.offset+lo, key, value

    def touched_pages(self, ids, pos, b):
        blocks = torch.cat((ids.reshape(-1), (pos//b-1).reshape(-1), (pos//b).reshape(-1)))
        tokens = blocks * b
        local = tokens[(tokens >= self.offset) & (tokens < self.offset+self.length)] - self.offset
        return (local//self.page_tokens).unique().cpu().tolist()


@torch.no_grad()
def sparse_attention(q, cache, positions, block_size, topk, backend="auto", group=None,
                     score_page_tokens=65536, return_selection=False):
    """Exact score/top-k + gate + sparse pop; optional sequence-shard group.

    q and positions are replicated across the inference group. topk remains
    GLOBAL across all shards and the gate uses ALL visible remote blocks.
    No [B,Q,H,N] scores, [B,Q,G] masks, or [B,Q,H,K,D] gathers are allocated.
    """
    b = block_size
    if topk < 1 or score_page_tokens < b or cache.page_tokens % b or cache.offset % b:
        raise ValueError("topk positive; cache pages/offset must be block aligned")
    use_triton = backend != "torch" and triton_supported(q, b)
    if use_triton:
        try:
            from .kernels.sparse import block_scores, sparse_pop
        except ImportError:
            if backend == "triton":
                raise
            use_triton = False
    if backend == "triton" and not use_triton:
        raise ValueError("unsupported Triton sparse configuration")
    dtype = _work_dtype(q)
    scores = torch.full((*q.shape[:3], topk), -torch.inf, device=q.device, dtype=dtype)
    ids = torch.full(scores.shape, -1, device=q.device, dtype=torch.long)
    zr = torch.full(q.shape[:3], -torch.inf, device=q.device, dtype=dtype)
    tile = max(b, score_page_tokens//b*b)
    for offset, k, _ in cache.pages(q.device, keys_only=True):
        for lo in range(0, k.shape[1], tile):
            kc = k[:, lo:lo+tile]
            start = offset+lo
            s = (block_scores(q, kc, positions, b, start) if use_triton
                 else _score_torch(q, kc, positions, b, start))
            zr = torch.logaddexp(zr, s.logsumexp(-1))
            count = min(topk, s.shape[-1])
            sv, si = s.topk(count, -1)
            si = si + start//b
            candidates = torch.cat((scores, sv), -1)
            candidate_ids = torch.cat((ids, si), -1)
            scores, take = candidates.topk(topk, -1)
            ids = candidate_ids.gather(-1, take)
    distributed = group is not None and dist.is_initialized() and dist.get_world_size(group) > 1
    if distributed:
        world = dist.get_world_size(group)
        all_scores, all_ids = [torch.empty_like(scores) for _ in range(world)], [torch.empty_like(ids) for _ in range(world)]
        dist.all_gather(all_scores, scores.contiguous(), group=group)
        dist.all_gather(all_ids, ids.contiguous(), group=group)
        scores, take = torch.cat(all_scores, -1).topk(topk, -1)
        ids = torch.cat(all_ids, -1).gather(-1, take)
        maximum = zr.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        safe = torch.where(torch.isfinite(maximum), maximum, 0.)
        mass = torch.exp(zr-safe)
        dist.all_reduce(mass, group=group)
        zr = safe + mass.log()
    ids = ids.masked_fill(~torch.isfinite(scores), -1)
    out = torch.zeros(q.shape, device=q.device, dtype=dtype)
    lse = torch.full(q.shape[:3], -torch.inf, device=q.device, dtype=dtype)
    for offset, k, v in cache.pages(q.device, cache.touched_pages(ids, positions, b)):
        if use_triton:
            part, pl = sparse_pop(q, k, v, positions, ids, scores, zr, b, offset)
        else:
            part, pl = _pop_torch(q, k, v, positions, ids, scores, zr, b, offset)
        out, lse = _merge_read(out, lse, part, pl)
    if distributed:
        maximum = lse.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        safe = torch.where(torch.isfinite(maximum), maximum, 0.)
        weight = torch.exp(lse-safe)
        out = out*weight[..., None]
        dist.all_reduce(out, group=group)
        dist.all_reduce(weight, group=group)
        out = out/weight.clamp_min(torch.finfo(dtype).tiny)[..., None]
    if return_selection:
        return out.to(q.dtype), {"indices": ids, "scores": scores, "log_zr": zr}
    return out.to(q.dtype)
