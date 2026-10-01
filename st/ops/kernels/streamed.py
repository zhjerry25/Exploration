"""Exact sub-block Triton reductions: no matrix staging or QxN workspace.

T is independent of model block B. Each complete block is reduced before
applying the gate; backward has unique key/query owners and no atomic writes.
"""
import triton as tr
import triton.language as tl
from .launch import launch_streamed


@tr.jit
def _add(x, y):
    m = tl.maximum(x, y)
    safe = tl.where(m == -float('inf'), 0., m)
    return safe+tl.log(tl.exp(x-safe)+tl.exp(y-safe))


@tr.jit
def _mix(s, mean, mass, value):
    total = _add(mass, s)
    safe = tl.where(total == -float('inf'), 0., total)
    return total, value*tl.exp(mass-safe)+mean*tl.exp(s-safe)


@tr.jit
def _stats(q, K, V, batch, head, start, end, SK: tl.constexpr,
           H: tl.constexpr, D: tl.constexpr, B: tl.constexpr, DM: tl.constexpr,
           T: tl.constexpr, SCALE: tl.constexpr):
    d, r = tl.arange(0, DM), tl.arange(0, T)
    m, den = -float('inf'), 0.
    num = tl.full((DM,), 0., tl.float32)
    for part in range(tl.cdiv(B, T)):
        token = start+part*T+r
        valid = (token < start+B) & (token < end)
        ptr = ((batch.to(tl.int64)*SK+token)*H+head)*D
        mask = valid[:, None] & (d[None, :] < D)
        key = tl.load(K+ptr[:, None]+d[None, :], mask, 0).to(tl.float32)
        a = tl.where(valid, tl.sum(key*q[None, :], 1)*SCALE, -float('inf'))
        next_m = tl.maximum(m, tl.max(a, 0))
        safe = tl.where(next_m == -float('inf'), 0., next_m)
        alpha, weight = tl.exp(m-safe), tl.exp(a-safe)
        value = tl.load(V+ptr[:, None]+d[None, :], mask, 0).to(tl.float32)
        num = num*alpha+tl.sum(weight[:, None]*value, 0)
        den = den*alpha+tl.sum(weight, 0)
        m = next_m
    return m+tl.log(den), num/tl.maximum(den, 1.e-30)


@tr.jit
def _derivative(q, go, key, value, valid, s, mean, remote, zr, z, delta, rho, SCALE: tl.constexpr):
    a = tl.where(valid, tl.sum(key*q[None, :], 1)*SCALE, -float('inf'))
    bias = tl.where(remote, s-zr, 0.)
    p = tl.exp(a+bias-z)
    direct = p*(tl.sum(value*go[None, :], 1)-delta)
    block_grad = tl.exp(s+bias-z)*(tl.sum(mean*go, 0)-delta)
    gate = tl.exp(a-s)*block_grad-tl.exp(a-zr)*rho
    return (direct+tl.where(remote, gate, 0.))*SCALE, p


@tr.jit
def _stream_forward(Q, K, V, POS, O, RC, PR, ZR, Z,
                    SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    B: tl.constexpr, DM: tl.constexpr, T: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head, query, batch = row % H, (row//H) % SQ, row//(H*SQ)
    d = tl.arange(0, DM)
    q = tl.load(Q+row*D+d, d < D, 0).to(tl.float32)
    pos = tl.load(POS+batch*SQ+query)
    end = tl.minimum(SK, pos+1)
    lm, rm, zr = -float('inf'), -float('inf'), -float('inf')
    lv, rv = tl.full((DM,), 0., tl.float32), tl.full((DM,), 0., tl.float32)
    for block in range(tl.cdiv(end, B)):
        s, mean = _stats(q, K, V, batch, head, block*B, end, SK, H, D, B, DM, T, SCALE)
        if block < pos//B-1:
            zr = _add(zr, s)
            rm, rv = _mix(2*s, mean, rm, rv)
        else:
            lm, lv = _mix(s, mean, lm, lv)
    rm = rm-tl.where(zr == -float('inf'), 0., zr)
    z = _add(lm, rm)
    safe = tl.where(z == -float('inf'), 0., z)
    pr = tl.exp(rm-safe)
    rc = rv*pr
    tl.store(O+row*D+d, lv*tl.exp(lm-safe)+rc, d < D)
    tl.store(RC+row*D+d, rc, d < D)
    tl.store(PR+row, pr)
    tl.store(ZR+row, zr)
    tl.store(Z+row, z)


@tr.jit
def _stream_backward_q(Q, K, V, POS, DO, ZR, Z, DELTA, RHO, DQ,
                       SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                       B: tl.constexpr, DM: tl.constexpr, T: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head, query, batch = row % H, (row//H) % SQ, row//(H*SQ)
    d, r = tl.arange(0, DM), tl.arange(0, T)
    q = tl.load(Q+row*D+d, d < D, 0).to(tl.float32)
    go = tl.load(DO+row*D+d, d < D, 0).to(tl.float32)
    pos = tl.load(POS+batch*SQ+query)
    zr, z = tl.load(ZR+row), tl.load(Z+row)
    zr, z = tl.where(zr == -float('inf'), 0., zr), tl.where(z == -float('inf'), 0., z)
    delta, rho = tl.load(DELTA+row), tl.load(RHO+row)
    end = tl.minimum(SK, pos+1)
    dq = tl.full((DM,), 0., tl.float32)
    for block in range(tl.cdiv(end, B)):
        start = block*B
        s, mean = _stats(q, K, V, batch, head, start, end, SK, H, D, B, DM, T, SCALE)
        for part in range(tl.cdiv(B, T)):
            token = start+part*T+r
            valid = (token < start+B) & (token < end)
            ptr = ((batch*SK+token)*H+head)*D
            mask = valid[:, None] & (d[None, :] < D)
            key = tl.load(K+ptr[:, None]+d[None, :], mask, 0).to(tl.float32)
            value = tl.load(V+ptr[:, None]+d[None, :], mask, 0).to(tl.float32)
            da, p = _derivative(q, go, key, value, valid, s, mean, block < pos//B-1, zr, z, delta, rho, SCALE)
            dq += tl.sum(da[:, None]*key, 0)
    tl.store(DQ+row*D+d, dq, d < D)


@tr.jit
def _stream_backward_kv(Q, K, V, POS, DO, ZR, Z, DELTA, RHO, DK, DV,
                        SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                        B: tl.constexpr, DM: tl.constexpr, T: tl.constexpr, SCALE: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    block, part = tile//tl.cdiv(B, T), tile % tl.cdiv(B, T)
    start, batch, head = block*B, bh//H, bh % H
    d, r = tl.arange(0, DM), tl.arange(0, T)
    token = start+part*T+r
    owned = (token < start+B) & (token < SK)
    ptr = ((batch*SK+token)*H+head)*D
    mask = owned[:, None] & (d[None, :] < D)
    key = tl.load(K+ptr[:, None]+d[None, :], mask, 0).to(tl.float32)
    value = tl.load(V+ptr[:, None]+d[None, :], mask, 0).to(tl.float32)
    dk, dv = tl.full((T, DM), 0., tl.float32), tl.full((T, DM), 0., tl.float32)
    for query in range(SQ):
        row = (batch*SQ+query)*H+head
        pos = tl.load(POS+batch*SQ+query)
        if pos >= start:
            q = tl.load(Q+row*D+d, d < D, 0).to(tl.float32)
            go = tl.load(DO+row*D+d, d < D, 0).to(tl.float32)
            zr, z = tl.load(ZR+row), tl.load(Z+row)
            zr = tl.where(zr == -float('inf'), 0., zr)
            delta, rho = tl.load(DELTA+row), tl.load(RHO+row)
            s, mean = _stats(q, K, V, batch, head, start, tl.minimum(SK, pos+1), SK, H, D, B, DM, T, SCALE)
            da, p = _derivative(q, go, key, value, owned & (token <= pos), s, mean, block < pos//B-1, zr, z, delta, rho, SCALE)
            dk += da[:, None]*q[None, :]
            dv += p[:, None]*go[None, :]
    tl.store(DK+ptr[:, None]+d[None, :], dk, mask)
    tl.store(DV+ptr[:, None]+d[None, :], dv, mask)


@tr.jit
def _stream_scores(Q, K, POS, S, OFFSET,
                   SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   G: tl.constexpr, B: tl.constexpr, DM: tl.constexpr, T: tl.constexpr, SCALE: tl.constexpr):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    head, query, batch = row % H, (row//H) % SQ, row//(H*SQ)
    d, r = tl.arange(0, DM), tl.arange(0, T)
    pos = tl.load(POS+batch*SQ+query)
    s = -float('inf')
    if (pos >= 0) & (block+OFFSET//B < pos//B-1) & ((block+1)*B <= SK):
        q = tl.load(Q+row*D+d, d < D, 0).to(tl.float32)
        for part in range(tl.cdiv(B, T)):
            token = block*B+part*T+r
            valid = token < (block+1)*B
            ptr = ((batch*SK+token)*H+head)*D
            key = tl.load(K+ptr[:, None]+d[None, :], valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
            a = tl.where(valid, tl.sum(key*q[None, :], 1)*SCALE, -float('inf'))
            m = tl.max(a, 0)
            s = _add(s, m+tl.log(tl.sum(tl.exp(a-m), 0)))
    tl.store(S+row*G+block, s)


def run_streamed(operation, args, meta, batch, device, dtype):
    options = {k: v for k, v in meta.items() if k not in ('M', 'N')}
    kernel = {'forward': _stream_forward, 'dq': _stream_backward_q,
              'dkv': _stream_backward_kv, 'scores': _stream_scores}[operation]
    if operation == 'dkv':
        grid = lambda c: (tr.cdiv(c['SK'], c['B'])*tr.cdiv(c['B'], c['T']), batch*c['H'])
    elif operation == 'scores':
        grid = lambda c: (batch*c['SQ']*c['H'], c['G'])
    else:
        grid = lambda c: (batch*c['SQ']*c['H'],)
    return launch_streamed(kernel, grid, args, options, device, dtype)
