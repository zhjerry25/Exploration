"""CUDA event timings and peak memory for exact dense/sparse operators.

Use st.run train for end-to-end, multi-GPU optimizer-step throughput. This
microbenchmark measures the attention operator only, including dense backward.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .attention import dense_attention
from .memory import GiB
from .sparse import TensorPages, sparse_attention
from .validate import environment


def dense_eager(q, k, v, positions, block_size):
    """Vectorized original dense algebra; intentionally materializes Q x N."""
    n, b = k.shape[1], block_size
    a = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    padded = torch.nn.functional.pad(a, (0, (-n) % b), value=-torch.inf)
    s = padded.reshape(*a.shape[:-1], -1, b).logsumexp(-1)
    ids = torch.arange(s.shape[-1], device=q.device)
    visible = ids[None, None, None, :] < positions[:, None, :, None]//b-1
    s = s.masked_fill(~visible, -torch.inf)
    has_remote = visible.any(-1, keepdim=True)
    # Avoid all--inf logsumexp in the differentiable graph.
    safe_s = torch.where(has_remote, s, torch.zeros_like(s))
    gate = (safe_s-safe_s.logsumexp(-1, keepdim=True)).masked_fill(~visible, 0.)
    bias = gate.repeat_interleave(b, -1)[..., :n]
    causal = torch.arange(n, device=q.device)[None, None, None, :] <= positions[:, None, :, None]
    probs = (a+bias).masked_fill(~causal, -torch.inf).softmax(-1)
    return (probs @ v.transpose(1, 2).float()).transpose(1, 2).to(q.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operation", choices=["dense", "sparse"], default="dense")
    p.add_argument("--backend", choices=["triton", "torch", "eager"], default="triton")
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--queries", type=int, default=0, help="0 = all for dense, 16 for sparse")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--page-tokens", type=int, default=65536)
    p.add_argument("--output", default="")
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("benchmark requires a remote NVIDIA GPU")
    if min(args.length, args.batch_size, args.heads, args.head_dim, args.block_size, args.iterations, args.topk) < 1 or args.warmup < 0:
        p.error("dimensions/iterations must be positive, warmup nonnegative")
    queries = args.queries or (args.length if args.operation == "dense" else min(16, args.length))
    if args.operation == "sparse" and args.backend == "eager":
        p.error("use torch as the paged sparse reference")
    if args.backend == "eager":
        free, _ = torch.cuda.mem_get_info()
        if args.batch_size*queries*args.heads*args.length*4*12 > free*.7:
            p.error("eager QxN estimate exceeds GPU budget; reduce length/queries")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    q = torch.randn(args.batch_size, queries, args.heads, args.head_dim, device="cuda", dtype=dtype, requires_grad=args.operation == "dense")
    k = torch.randn(args.batch_size, args.length, args.heads, args.head_dim, device="cuda", dtype=dtype, requires_grad=args.operation == "dense")
    v = torch.randn_like(k, requires_grad=args.operation == "dense")
    positions = torch.linspace(0 if args.operation == "dense" else args.length-queries,
                               args.length-1, queries, device="cuda").round().long()[None].expand(args.batch_size, -1).contiguous()
    probe = torch.randn_like(q)
    cache = TensorPages(k, v, max(args.block_size, args.page_tokens//args.block_size*args.block_size))

    def iteration():
        if args.operation == "dense":
            q.grad = k.grad = v.grad = None
            fn = (lambda: dense_eager(q, k, v, positions, args.block_size)) if args.backend == "eager" else (
                lambda: dense_attention(q, k, v, positions, args.block_size, args.backend))
            out = fn()
            out.backward(probe)
        else:
            out = sparse_attention(q, cache, positions, args.block_size, args.topk, args.backend)
        return out

    for _ in range(args.warmup):
        iteration()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iterations):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        iteration()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end))
    ordered = sorted(times)
    report = {"environment": environment(), "config": vars(args), "queries": queries,
              "latency_ms_median": ordered[len(ordered)//2], "latency_ms_min": min(times),
              "latency_ms_samples": times,
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/GiB,
              "peak_reserved_gib": torch.cuda.max_memory_reserved()/GiB}
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
