"""Fused exact dense block-gated attention, including the gate derivative.

Forward stores O(B Q H D), backward recomputes tiles. dQ and dK/dV have
separate owners, avoiding fp32 atomics and an O(Q N) allocation. No dropout.
"""
import torch
import triton as tr
import triton.language as tl
from .launch import ResourceExhausted, launch


@tr.jit
def _block_lse(a, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    ab = tl.reshape(a, (M, N // B, B))
    mx = tl.max(ab, 2)
    safe = tl.where(mx == -float("inf"), 0., mx)
    s = safe + tl.log(tl.sum(tl.exp(ab - safe[:, :, None]), 2))
    return tl.reshape(tl.broadcast_to(s[:, :, None], (M, N // B, B)), (M, N))


@tr.jit
def _online(a, v, m, den, num):
    m1 = tl.maximum(m, tl.max(a, 1))
    safe = tl.where(m1 == -float("inf"), 0., m1)
    alpha = tl.exp(m - safe)
    p = tl.exp(a - safe[:, None])
    den = den * alpha + tl.sum(p, 1)
    num = num * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
    return m1, den, num


@tr.jit
def _forward(Q, K, V, POS, O, RC, PR, ZR, Z,
             SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             B: tl.constexpr, DM: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
             SCALE: tl.constexpr):
    qi = tl.program_id(0) * M + tl.arange(0, M)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    d = tl.arange(0, DM)
    q = tl.load(Q + ((batch * SQ + qi[:, None]) * H + head) * D + d[None, :],
                (qi[:, None] < SQ) & (d[None, :] < D), 0)
    pos = tl.load(POS + batch * SQ + qi, qi < SQ, -1)
    ml = tl.full((M,), -float("inf"), tl.float32)
    mr = tl.full((M,), -float("inf"), tl.float32)
    zr = tl.full((M,), -float("inf"), tl.float32)
    dl, dr = tl.full((M,), 0., tl.float32), tl.full((M,), 0., tl.float32)
    nl, nr = tl.full((M, DM), 0., tl.float32), tl.full((M, DM), 0., tl.float32)
    limit = tl.minimum(SK, tl.max(pos, 0) + 1)
    for start in range(0, tl.cdiv(limit, N)):
        ki = start * N + tl.arange(0, N)
        ptr = ((batch * SK + ki) * H + head) * D
        k = tl.load(K + ptr[None, :] + d[:, None], (ki[None, :] < SK) & (d[:, None] < D), 0)
        v = tl.load(V + ptr[:, None] + d[None, :], (ki[:, None] < SK) & (d[None, :] < D), 0)
        a = tl.dot(q, k, input_precision="ieee") * SCALE
        a = tl.where(ki[None, :] < SK, a, -float("inf"))
        sb = _block_lse(a, M, N, B)
        remote = (ki[None, :] // B < pos[:, None] // B - 1) & (pos[:, None] >= 0)
        local = ((ki[None, :] // B >= pos[:, None] // B - 1)
                 & (ki[None, :] <= pos[:, None]) & (pos[:, None] >= 0))
        ar = tl.where(remote, a, -float("inf"))
        tm = tl.max(ar, 1)
        safe_tm = tl.where(tm == -float("inf"), 0., tm)
        tz = safe_tm + tl.log(tl.sum(tl.exp(ar - safe_tm[:, None]), 1))
        zm = tl.maximum(zr, tz)
        safe_zm = tl.where(zm == -float("inf"), 0., zm)
        zr = safe_zm + tl.log(tl.exp(zr - safe_zm) + tl.exp(tz - safe_zm))
        mr, dr, nr = _online(tl.where(remote, a + sb, -float("inf")), v, mr, dr, nr)
        ml, dl, nl = _online(tl.where(local, a, -float("inf")), v, ml, dl, nl)
    lr = mr + tl.log(dr) - tl.where(zr == -float("inf"), 0., zr)
    ll = ml + tl.log(dl)
    mx = tl.maximum(ll, lr)
    safe_mx = tl.where(mx == -float("inf"), 0., mx)
    total = safe_mx + tl.log(tl.exp(ll - safe_mx) + tl.exp(lr - safe_mx))
    safe_total = tl.where(total == -float("inf"), 0., total)
    pr = tl.exp(lr - safe_total)
    rc = nr / tl.maximum(dr[:, None], 1.e-30) * pr[:, None]
    out = rc + nl / tl.maximum(dl[:, None], 1.e-30) * tl.exp(ll - safe_total)[:, None]
    op = ((batch * SQ + qi[:, None]) * H + head) * D + d[None, :]
    mask = (qi[:, None] < SQ) & (d[None, :] < D)
    tl.store(O + op, out, mask)
    tl.store(RC + op, rc, mask)
    sp = (batch * SQ + qi) * H + head
    tl.store(PR + sp, pr, qi < SQ)
    tl.store(ZR + sp, zr, qi < SQ)
    tl.store(Z + sp, total, qi < SQ)


@tr.jit
def _preprocess(O, RC, PR, DO, DELTA, RHO, TOTAL: tl.constexpr, D: tl.constexpr, DM: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, DM)
    out = tl.load(O + row * D + d, d < D, 0).to(tl.float32)
    rc = tl.load(RC + row * D + d, d < D, 0)
    go = tl.load(DO + row * D + d, d < D, 0).to(tl.float32)
    pr = tl.load(PR + row)
    delta = tl.sum(out * go, 0)
    rho = tl.sum(rc * go, 0) - delta * pr
    tl.store(DELTA + row, delta)
    tl.store(RHO + row, rho)


@tr.jit
def _derivative(q, k, v, go, pos, ki, zr, z, delta, rho,
                SK: tl.constexpr, B: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                SCALE: tl.constexpr):
    a = tl.dot(q, k, input_precision="ieee") * SCALE
    a = tl.where(ki[None, :] < SK, a, -float("inf"))
    sb = _block_lse(a, M, N, B)
    remote = (ki[None, :] // B < pos[:, None] // B - 1) & (pos[:, None] >= 0)
    local = ((ki[None, :] // B >= pos[:, None] // B - 1)
             & (ki[None, :] <= pos[:, None]) & (pos[:, None] >= 0))
    p = tl.exp(tl.where(remote, a + sb - zr[:, None], a) - z[:, None])
    p = tl.where(remote | local, p, 0.)
    direct = p * (tl.dot(go, tl.trans(v), input_precision="ieee") - delta[:, None])
    rb = tl.sum(tl.reshape(direct, (M, N // B, B)), 2)
    rb = tl.reshape(tl.broadcast_to(rb[:, :, None], (M, N // B, B)), (M, N))
    gate = tl.exp(a - sb) * rb - tl.exp(a - zr[:, None]) * rho[:, None]
    da = (direct + tl.where(remote, gate, 0.)) * SCALE
    return da, p


@tr.jit
def _backward_q(Q, K, V, POS, DO, ZR, Z, DELTA, RHO, DQ,
                SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                B: tl.constexpr, DM: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                SCALE: tl.constexpr):
    qi = tl.program_id(0) * M + tl.arange(0, M)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    d = tl.arange(0, DM)
    qp = ((batch * SQ + qi[:, None]) * H + head) * D + d[None, :]
    qm = (qi[:, None] < SQ) & (d[None, :] < D)
    q, go = tl.load(Q + qp, qm, 0), tl.load(DO + qp, qm, 0)
    pos = tl.load(POS + batch * SQ + qi, qi < SQ, -1)
    sp = (batch * SQ + qi) * H + head
    zr, z = tl.load(ZR + sp, qi < SQ, 0), tl.load(Z + sp, qi < SQ, 0)
    zr, z = tl.where(zr == -float("inf"), 0., zr), tl.where(z == -float("inf"), 0., z)
    delta = tl.load(DELTA + sp, qi < SQ, 0)
    rho = tl.load(RHO + sp, qi < SQ, 0)
    dq = tl.full((M, DM), 0., tl.float32)
    limit = tl.minimum(SK, tl.max(pos, 0) + 1)
    for start in range(0, tl.cdiv(limit, N)):
        ki = start * N + tl.arange(0, N)
        kp = ((batch * SK + ki) * H + head) * D
        k = tl.load(K + kp[None, :] + d[:, None], (ki[None, :] < SK) & (d[:, None] < D), 0)
        v = tl.load(V + kp[:, None] + d[None, :], (ki[:, None] < SK) & (d[None, :] < D), 0)
        da, p = _derivative(q, k, v, go, pos, ki, zr, z, delta, rho, SK, B, M, N, SCALE)
        dq += tl.dot(da.to(k.dtype), tl.trans(k), input_precision="ieee")
    tl.store(DQ + qp, dq, qm)


@tr.jit
def _backward_kv(Q, K, V, POS, DO, ZR, Z, DELTA, RHO, DK, DV,
                 SQ: tl.constexpr, SK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 B: tl.constexpr, DM: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                 SCALE: tl.constexpr):
    ki = tl.program_id(0) * N + tl.arange(0, N)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    d = tl.arange(0, DM)
    kp = ((batch * SK + ki) * H + head) * D
    km = (ki[:, None] < SK) & (d[None, :] < D)
    kt = tl.load(K + kp[:, None] + d[None, :], km, 0)
    v = tl.load(V + kp[:, None] + d[None, :], km, 0)
    dk, dv = tl.full((N, DM), 0., tl.float32), tl.full((N, DM), 0., tl.float32)
    for start in range(0, tl.cdiv(SQ, M)):
        qi = start * M + tl.arange(0, M)
        qp = ((batch * SQ + qi[:, None]) * H + head) * D + d[None, :]
        qm = (qi[:, None] < SQ) & (d[None, :] < D)
        q, go = tl.load(Q + qp, qm, 0), tl.load(DO + qp, qm, 0)
        pos = tl.load(POS + batch * SQ + qi, qi < SQ, -1)
        sp = (batch * SQ + qi) * H + head
        zr, z = tl.load(ZR + sp, qi < SQ, 0), tl.load(Z + sp, qi < SQ, 0)
        zr, z = tl.where(zr == -float("inf"), 0., zr), tl.where(z == -float("inf"), 0., z)
        delta = tl.load(DELTA + sp, qi < SQ, 0)
        rho = tl.load(RHO + sp, qi < SQ, 0)
        da, p = _derivative(q, tl.trans(kt), v, go, pos, ki, zr, z, delta, rho, SK, B, M, N, SCALE)
        dk += tl.dot(tl.trans(da.to(q.dtype)), q, input_precision="ieee")
        dv += tl.dot(tl.trans(p.to(go.dtype)), go, input_precision="ieee")
    tl.store(DK + kp[:, None] + d[None, :], dk, km)
    tl.store(DV + kp[:, None] + d[None, :], dv, km)


def _dispatch(operation, force_streamed, kernel, grid, args, meta, batch, device, dtype):
    if not force_streamed:
        try:
            return launch(kernel, grid, args, meta, device, dtype)
        except ResourceExhausted:
            pass  # Resource failures occur before execution; errors of other kinds propagate.
    from .streamed import run_streamed
    return run_streamed(operation, args, meta, batch, device, dtype)


class _Dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos, block_size, force_streamed):
        q, k, v, pos = (t.contiguous() for t in (q, k, v, pos))
        batch, queries, heads, dim = q.shape
        out = torch.empty_like(q)
        rc = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        pr = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)
        zr, z = torch.empty_like(pr), torch.empty_like(pr)
        tile_n, tile_m = max(64, block_size), 16
        meta = dict(SQ=queries, SK=k.shape[1], H=heads, D=dim, B=block_size,
                    DM=max(16, tr.next_power_of_2(dim)), M=tile_m, N=tile_n,
                    SCALE=dim ** -0.5)
        _dispatch("forward", force_streamed, _forward, lambda c: (tr.cdiv(queries, c["M"]), batch*heads),
               (q, k, v, pos, out, rc, pr, zr, z), meta, batch, q.device, q.dtype)
        ctx.force_streamed = force_streamed
        ctx.save_for_backward(q, k, v, pos, out, rc, pr, zr, z)
        ctx.meta = meta
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        q, k, v, pos, out, rc, pr, zr, z = ctx.saved_tensors
        grad = grad.contiguous()
        delta, rho = torch.empty_like(pr), torch.empty_like(pr)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        batch, queries, heads, dim = q.shape
        _preprocess[(batch * queries * heads,)](out, rc, pr, grad, delta, rho,
            TOTAL=batch * queries * heads, D=dim, DM=tr.next_power_of_2(dim))
        _dispatch("dq", ctx.force_streamed, _backward_q, lambda c: (tr.cdiv(queries, c["M"]), batch*heads),
               (q, k, v, pos, grad, zr, z, delta, rho, dq), ctx.meta, batch, q.device, q.dtype)
        _dispatch("dkv", ctx.force_streamed, _backward_kv, lambda c: (tr.cdiv(k.shape[1], c["N"]), batch*heads),
               (q, k, v, pos, grad, zr, z, delta, rho, dk, dv), ctx.meta, batch, q.device, q.dtype)
        return dq, dk, dv, None, None, None


def dense_triton(q, k, v, positions, block_size, *, force_streamed=False):
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must use the same dtype")
    return _Dense.apply(q, k, v, positions, block_size, force_streamed)
