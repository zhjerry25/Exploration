"""Exact gated attention with bounded workspace.

The remote gate is log M_j - log Z_R, M_j = sum_{i in j} exp(q.k_i).
It is NOT ordinary FlashAttention. Both gate terms participate in backward.
The torch implementation is a portable, double precision reference and a
bounded-memory fallback. CUDA dispatch is lazy; importing st needs no Triton.
"""
import math

import torch


def _work_dtype(t):
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _tile(q, k, pos, offset, block_size):
    # q [B,H,Q,D], k [B,H,K,D], K is block aligned, tail padded.
    a = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    b, nk = block_size, k.shape[-2]
    t = torch.arange(offset, offset + nk, device=q.device)
    group = t // b
    valid_q = pos[:, None, :, None] >= 0
    remote = (group[None, None, None, :] <
              pos[:, None, :, None] // b - 1) & valid_q
    local = ((group[None, None, None, :] >= pos[:, None, :, None] // b - 1)
             & (t[None, None, None, :] <= pos[:, None, :, None]) & valid_q)
    s = a.reshape(*a.shape[:-1], nk // b, b).logsumexp(-1)
    sb = s.repeat_interleave(b, -1)
    return a, sb, remote, local


def _kv_tile(k, v, start, end, block_size, dtype, device):
    pad = (-(end - start)) % block_size
    kc = k[:, start:end].transpose(1, 2).to(dtype)
    vc = v[:, start:end].transpose(1, 2).to(dtype)
    if pad:
        kc = torch.nn.functional.pad(kc, (0, 0, 0, pad))
        vc = torch.nn.functional.pad(vc, (0, 0, 0, pad))
    valid = torch.arange(start, start + kc.shape[-2], device=device) < k.shape[1]
    return kc, vc, valid


def _merge(logits, v, m, den, num):
    new_m = torch.maximum(m, logits.amax(-1))
    safe = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
    alpha = torch.exp(m - safe)
    p = torch.exp(logits - safe[..., None])
    return new_m, den * alpha + p.sum(-1), num * alpha[..., None] + p @ v


def _forward_torch(q, k, v, positions, block_size, q_chunk, kv_chunk):
    dtype = _work_dtype(q)
    batch, queries, heads, dim = q.shape
    out = torch.empty_like(q)
    rem_out = torch.empty(q.shape, device=q.device, dtype=dtype)
    pr = torch.empty((batch, queries, heads), device=q.device, dtype=dtype)
    lzr, lse = torch.empty_like(pr), torch.empty_like(pr)
    nk = k.shape[1]
    kv_chunk = max(block_size, kv_chunk // block_size * block_size)
    # One tile covering K: cast/pad it once instead of per query chunk.
    held = _kv_tile(k, v, 0, nk, block_size, dtype, q.device) if nk <= kv_chunk else None
    for lo in range(0, queries, q_chunk):
        hi = min(lo + q_chunk, queries)
        qc = q[:, lo:hi].transpose(1, 2).to(dtype)
        pos = positions[:, lo:hi]
        shape = qc.shape[:-1]
        ml = torch.full(shape, -torch.inf, dtype=dtype, device=q.device)
        mr, zr = ml.clone(), ml.clone()
        dl, dr = torch.zeros_like(ml), torch.zeros_like(ml)
        nl, nr = torch.zeros_like(qc), torch.zeros_like(qc)
        for start in range(0, nk, kv_chunk):
            end = min(start + kv_chunk, nk)
            kc, vc, valid = held if held is not None else _kv_tile(k, v, start, end, block_size, dtype, q.device)
            a, sb, remote, local = _tile(qc, kc, pos, start, block_size)
            # Invalid padded keys cannot be remote (positions are < nk).
            local = local & valid
            zr = torch.logaddexp(zr, a.masked_fill(~remote, -torch.inf).logsumexp(-1))
            mr, dr, nr = _merge((a + sb).masked_fill(~remote, -torch.inf), vc, mr, dr, nr)
            ml, dl, nl = _merge(a.masked_fill(~local, -torch.inf), vc, ml, dl, nl)
        lr = mr + dr.log() - torch.where(torch.isfinite(zr), zr, 0.)
        ll = ml + dl.log()
        total = torch.logaddexp(lr, ll)
        safe_total = torch.where(torch.isfinite(total), total, 0.)
        rp = torch.exp(lr - safe_total)
        rc = nr / dr.clamp_min(torch.finfo(dtype).tiny)[..., None] * rp[..., None]
        lc = nl / dl.clamp_min(torch.finfo(dtype).tiny)[..., None] * torch.exp(ll - safe_total)[..., None]
        out[:, lo:hi] = (lc + rc).transpose(1, 2).to(q.dtype)
        rem_out[:, lo:hi] = rc.transpose(1, 2)
        pr[:, lo:hi] = rp.transpose(1, 2)
        lzr[:, lo:hi] = zr.transpose(1, 2)
        lse[:, lo:hi] = total.transpose(1, 2)
    return out, rem_out, pr, lzr, lse


class _TorchDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, positions, block_size, q_chunk, kv_chunk):
        out, rem, pr, lzr, lse = _forward_torch(q, k, v, positions, block_size, q_chunk, kv_chunk)
        ctx.save_for_backward(q, k, v, positions, out, rem, pr, lzr, lse)
        ctx.block_size, ctx.q_chunk, ctx.kv_chunk = block_size, q_chunk, kv_chunk
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        q, k, v, positions, out, rem, pr, lzr, lse = ctx.saved_tensors
        dtype = _work_dtype(q)
        dq, dk, dv = (torch.zeros(t.shape, device=t.device, dtype=dtype) for t in (q, k, v))
        b, nk = ctx.block_size, k.shape[1]
        kc_size = max(b, ctx.kv_chunk // b * b)
        held = _kv_tile(k, v, 0, nk, b, dtype, q.device) if nk <= kc_size else None
        for lo in range(0, q.shape[1], ctx.q_chunk):
            hi = min(lo + ctx.q_chunk, q.shape[1])
            qc = q[:, lo:hi].transpose(1, 2).to(dtype)
            gc = grad[:, lo:hi].transpose(1, 2).to(dtype)
            oc = out[:, lo:hi].transpose(1, 2).to(dtype)
            delta = (gc * oc).sum(-1)
            rho = (gc * rem[:, lo:hi].transpose(1, 2)).sum(-1) - delta * pr[:, lo:hi].transpose(1, 2)
            zr = lzr[:, lo:hi].transpose(1, 2)
            zr = torch.where(torch.isfinite(zr), zr, 0.)
            z = lse[:, lo:hi].transpose(1, 2)
            z = torch.where(torch.isfinite(z), z, 0.)
            for start in range(0, nk, kc_size):
                end = min(start + kc_size, nk)
                kc, vc, valid = held if held is not None else _kv_tile(k, v, start, end, b, dtype, q.device)
                a, sb, remote, local = _tile(qc, kc, positions[:, lo:hi], start, b)
                local = local & valid
                probs = torch.exp(torch.where(remote, a + sb - zr[..., None], a) - z[..., None])
                probs = torch.where(remote | local, probs, 0.)
                direct = probs * (gc @ vc.transpose(-1, -2) - delta[..., None])
                block_grad = direct.reshape(*direct.shape[:-1], -1, b).sum(-1).repeat_interleave(b, -1)
                # Exact derivative through BOTH log M_j and log Z_R.
                gate_grad = torch.exp(a - sb) * block_grad - torch.exp(a - zr[..., None]) * rho[..., None]
                da = (direct + torch.where(remote, gate_grad, 0.)) / math.sqrt(q.shape[-1])
                dq[:, lo:hi] += (da @ kc).transpose(1, 2)
                dk[:, start:end] += (da.transpose(-1, -2) @ qc).transpose(1, 2)[:, :end-start]
                dv[:, start:end] += (probs.transpose(-1, -2) @ gc).transpose(1, 2)[:, :end-start]
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def triton_supported(q, block_size):
    return (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 2 <= block_size <= 128 and block_size & (block_size - 1) == 0
            and q.shape[-1] <= 256)


def dense_attention(q, k, v, positions, block_size, backend="auto", q_chunk=32, kv_chunk=1024):
    """[B,Q,H,D] x [B,N,H,D], positions [B,Q]; -1 denotes padding.

    No N x Q tensor is saved. First derivatives only. 'triton' fails loudly
    for unsupported layouts or missing Triton; auto falls back only for an
    unsupported configuration, never after a kernel/compilation error.
    """
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"unknown attention backend {backend!r}")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or positions.shape != q.shape[:2]:
        raise ValueError("expected q[B,Q,H,D], k/v[B,N,H,D], positions[B,Q]")
    if k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]:
        raise ValueError("query and KV batch/head dimensions differ")
    if min(q.shape) < 1 or k.shape[1] < 1 or min(block_size, q_chunk, kv_chunk) < 1:
        raise ValueError("attention dimensions and tile sizes must be positive")
    if backend != "torch" and triton_supported(q, block_size):
        try:
            from .kernels.dense import dense_triton
        except ImportError:
            if backend == "triton":
                raise
        else:
            return dense_triton(q, k, v, positions, block_size)
    elif backend == "triton":
        raise ValueError("Triton requires CUDA fp16/bf16/fp32, head_dim <=256 and power-of-two block_size 2..128")
    # Custom Function bodies execute outside autocast, so reference matmuls
    # accumulate in fp32 (or fp64 for numerical oracles).
    with torch.autocast(q.device.type, enabled=False):
        return _TorchDense.apply(q, k, v, positions, block_size, q_chunk, kv_chunk)
