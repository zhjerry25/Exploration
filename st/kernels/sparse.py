"""Exact block scores and gather-free sparse pop from one KV page/shard."""
import torch
import triton as tr
import triton.language as tl

from .dense import _block_lse
from .launch import launch


@tr.jit
def _scores(Q, K, POS, S, OFFSET, SQ: tl.constexpr, SK: tl.constexpr,
            H: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
            B: tl.constexpr, DM: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
            SCALE: tl.constexpr):
    qi = tl.program_id(0) * M + tl.arange(0, M)
    ki = tl.program_id(1) * N + tl.arange(0, N)
    bh = tl.program_id(2)
    batch, head = bh // H, bh % H
    d = tl.arange(0, DM)
    q = tl.load(Q + ((batch*SQ + qi[:, None])*H + head)*D + d[None, :],
                (qi[:, None] < SQ) & (d[None, :] < D), 0)
    k = tl.load(K + ((batch*SK + ki[None, :])*H + head)*D + d[:, None],
                (ki[None, :] < SK) & (d[:, None] < D), 0)
    pos = tl.load(POS + batch*SQ + qi, qi < SQ, -1)
    a = tl.dot(q, k, input_precision="ieee") * SCALE
    a = tl.where(ki[None, :] < SK, a, -float("inf"))
    ab = tl.reshape(a, (M, N//B, B))
    mx = tl.max(ab, 2)
    safe = tl.where(mx == -float("inf"), 0., mx)
    s = safe + tl.log(tl.sum(tl.exp(ab - safe[:, :, None]), 2))
    gi = tl.program_id(1)*(N//B) + tl.arange(0, N//B)
    visible = ((gi[None, :] + OFFSET//B < pos[:, None]//B-1)
               & (pos[:, None] >= 0) & ((gi[None, :]+1)*B <= SK))
    s = tl.where(visible, s, -float("inf"))
    tl.store(S + ((batch*SQ + qi[:, None])*H + head)*G + gi[None, :],
             s, (qi[:, None] < SQ) & (gi[None, :] < G))


@tr.jit
def _pop(Q, K, V, POS, IDS, SCORES, ZR, O, LSE, OFFSET,
         SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
         KEEP: tl.constexpr, B: tl.constexpr,
         DM: tl.constexpr, T: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    head, query, batch = row % H, (row//H) % SQ, row//(H*SQ)
    d = tl.arange(0, DM)
    r = tl.arange(0, T)
    q = tl.load(Q + row*D + d, d < D, 0).to(tl.float32)
    pos = tl.load(POS + batch*SQ + query)
    zr = tl.load(ZR + row)
    zr = tl.where(zr == -float("inf"), 0., zr)
    m, den = -float("inf"), 0.
    num = tl.full((DM,), 0., tl.float32)
    # Includes direct local tokens first, then selected blocks. Page masks
    # retain only this shard's keys, without constructing gathered K/V.
    for j in range(tl.cdiv((KEEP+2)*B, T)):
        slot = j*T + r
        is_local = slot < 2*B
        which = (slot-2*B)//B
        selected = tl.load(IDS + row*KEEP + which, (which >= 0) & (which < KEEP), -1)
        score = tl.load(SCORES + row*KEEP + which, (which >= 0) & (which < KEEP), -float("inf"))
        token = tl.where(is_local, (pos//B-1)*B + slot, selected*B + slot % B)
        valid = ((pos >= 0) & (token >= 0) & (token <= pos) & (token >= OFFSET)
                 & (token < OFFSET+SK) & (slot < (KEEP+2)*B)
                 & (is_local | ((selected >= 0) & (score != -float("inf")))))
        kp = ((batch*SK + token-OFFSET)*H + head)*D
        k = tl.load(K + kp[:, None] + d[None, :], valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
        v = tl.load(V + kp[:, None] + d[None, :], valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
        a = tl.sum(k*q[None, :], 1)*SCALE + tl.where(is_local, 0., score-zr)
        a = tl.where(valid, a, -float("inf"))
        m1 = tl.maximum(m, tl.max(a, 0))
        safe = tl.where(m1 == -float("inf"), 0., m1)
        alpha = tl.exp(m-safe)
        p = tl.exp(a-safe)
        num = num*alpha + tl.sum(p[:, None]*v, 0)
        den = den*alpha + tl.sum(p, 0)
        m = m1
    tl.store(O + row*D + d, num/tl.maximum(den, 1.e-30), d < D)
    tl.store(LSE + row, m+tl.log(den))


def block_scores(q, k, positions, block_size, offset=0):
    q, k, positions = (x.contiguous() for x in (q, k, positions))
    batch, queries, heads, dim = q.shape
    groups = tr.cdiv(k.shape[1], block_size)
    scores = torch.empty((batch, queries, heads, groups), device=q.device, dtype=torch.float32)
    m, n = 16, max(64, block_size)
    meta = dict(SQ=queries, SK=k.shape[1], H=heads, D=dim, G=groups,
                B=block_size, DM=max(16, tr.next_power_of_2(dim)),
                M=m, N=n, SCALE=dim**-0.5)
    launch(_scores, lambda c: (tr.cdiv(queries, c["M"]), tr.cdiv(k.shape[1], c["N"]), batch*heads),
           (q, k, positions, scores, offset), meta, q.device, q.dtype)
    return scores


def sparse_pop(q, k, v, positions, ids, scores, log_zr, block_size, offset=0):
    q, k, v, positions, ids, scores, log_zr = (
        t.contiguous() for t in (q, k, v, positions, ids, scores, log_zr))
    batch, queries, heads, dim = q.shape
    # Accumulate page/shard contributions in fp32, including for bf16 KV.
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    lse = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)
    _pop[(batch*queries*heads,)](q, k, v, positions, ids, scores, log_zr, out, lse, offset,
        SQ=queries, SK=k.shape[1], H=heads, D=dim, KEEP=ids.shape[-1],
        B=block_size, DM=tr.next_power_of_2(dim), T=32, SCALE=dim**-0.5, num_warps=4)
    return out, lse
