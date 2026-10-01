"""Remote validation: reference, CUDA kernel parity, and distributed parity.

Failures are fatal and recorded in JSON. There is no silent skip-to-success
when CUDA, Triton, or the requested distributed topology is unavailable.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .attention import dense_attention
from .baseline import BaselineModel
from .inference import InferenceSession
from .parallel import ParallelContext, sum_all
from .sparse import TensorPages, sparse_attention
from .stack_model import StackModel


def environment():
    record = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "platform": platform.platform()}
    if torch.cuda.is_available():
        record["gpus"] = [{"name": torch.cuda.get_device_name(i),
                           "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                           "capability": torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]
    try:
        import triton
        record["triton"] = triton.__version__
    except ImportError:
        record["triton"] = None
    return record


def reference_suite():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    if not (root/"tests").is_dir():
        raise RuntimeError("reference validation requires the source checkout, including tests/")
    suite = unittest.defaultTestLoader.discover(str(root/"tests"), top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(f"reference tests: {len(result.failures)} failures, {len(result.errors)} errors")
    return {"tests": result.testsRun, "skipped": len(result.skipped)}


def error_metrics(actual, expected):
    actual, expected = actual.detach(), expected.detach()
    difference = (actual.float()-expected.float()).abs()
    return {"max_abs": float(difference.max()),
            "relative_rms": float(difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1.e-8))}


def cuda_suite():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA suite requires an NVIDIA GPU")
    import triton  # required, not optional for this suite
    from tests.test_attention import eager_attention
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(772)
    results = []
    # Partial tails, empty remote sets, non-power-of-two D, all supported
    # block-size extremes, multiple batches/heads, and dummy query rows.
    cases = [(17, 2, 16), (129, 16, 32), (257, 32, 64), (193, 64, 96),
             (385, 128, 128), (131, 16, 256)]
    for dtype in (torch.float32, torch.bfloat16):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 support is required for the target validation matrix")
        for n, b, d in cases:
            print(json.dumps({"event": "case_start", "dtype": str(dtype), "n": n,
                              "block": b, "head_dim": d}), flush=True)
            q = torch.randn(2, 5, 2, d, device="cuda", dtype=dtype, requires_grad=True)
            k = torch.randn(2, n, 2, d, device="cuda", dtype=dtype, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            pos = torch.tensor([[0, b-1, min(2*b+1, n-1), n-1, -1], [n-1, 0, n//2, -1, n-2]], device="cuda")
            rq, rk, rv = [t.detach().float().requires_grad_() for t in (q, k, v)]
            expected = eager_attention(rq, rk, rv, pos, b)
            actual = dense_attention(q, k, v, pos, b, "triton")
            probe = torch.randn_like(actual)
            ag = torch.autograd.grad((actual*probe).sum(), (q, k, v))
            eg = torch.autograd.grad((expected*probe.float()).sum(), (rq, rk, rv))
            atol, rtol = ((2.e-5, 2.e-4) if dtype == torch.float32 else (3.e-2, 5.e-2))
            torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
            for a, e in zip(ag, eg):
                torch.testing.assert_close(a.float(), e, atol=atol, rtol=rtol)
            record = {"operation": "dense", "dtype": str(dtype), "n": n, "block": b, "head_dim": d,
                      "forward": error_metrics(actual, expected),
                      "gradients": [error_metrics(a, e) for a, e in zip(ag, eg)]}
            results.append(record)
            print(json.dumps(record), flush=True)
            with torch.no_grad():
                keep = 2
                expected_sparse = eager_attention(rq, rk, rv, pos, b, keep)
                sparse, selection = sparse_attention(q.detach(), TensorPages(k.detach(), v.detach(), 4*b),
                    pos, b, keep, "triton", score_page_tokens=2*b, return_selection=True)
                torch.testing.assert_close(sparse.float(), expected_sparse, atol=atol, rtol=rtol)
                record = {"operation": "sparse", "dtype": str(dtype), "n": n, "block": b,
                          "head_dim": d, **error_metrics(sparse, expected_sparse)}
                results.append(record)
                print(json.dumps(record), flush=True)
    # End-to-end weights, chunked CE and checkpoint recomputation.
    torch.manual_seed(42)
    model = StackModel(128, 64, 4, 16, 64, "Lx2,(G)x2", backend="triton",
                       checkpoint_chunks=True, encoder_chunk=32, pos_chunk=7).cuda()
    ref = copy.deepcopy(model)
    ref.backend = "torch"
    ids = torch.randint(128, (2, 131), device="cuda")
    targets = torch.randint(128, ids.shape, device="cuda")
    sup = torch.rand_like(ids, dtype=torch.float32) > .75
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(ids, sup=sup, targets=targets)["loss_sum"]
        expected = ref(ids, sup=sup, targets=targets)["loss_sum"]
    torch.testing.assert_close(actual, expected, atol=.1, rtol=.003)
    actual.backward()
    expected.backward()
    gradient_error = {}
    for (name, a), (_, e) in zip(model.named_parameters(), ref.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, e.grad, atol=.08, rtol=.06, msg=name)
            gradient_error[name] = error_metrics(a.grad, e.grad)
    results.append({"operation": "end_to_end_bf16", "loss": error_metrics(actual, expected), "gradients": gradient_error})
    from .kernels.launch import launch_report
    return {"cases": results, "kernel_launches": launch_report()}


def distributed_suite(cp_size):
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("distributed suite requires torchrun with >=2 NVIDIA GPUs")
    parallel = ParallelContext.initialize(cp_size, "cuda")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    from tests.test_attention import eager_attention
    records = []
    try:
        for model_kind in ("stack", "baseline"):
            for kind in ("full", "tail", "ragged"):
                torch.manual_seed(928)
                base = (StackModel(31, 32, 8, 4, 64, "(L)x2,(G)x2", backend="torch",
                                   checkpoint_chunks=True, encoder_chunk=8, pos_chunk=3)
                        if model_kind == "stack" else
                        BaselineModel(31, 32, 8, layers=2, checkpoint_chunks=True, loss_chunk=3))
                base = base.to(device).double()
                ref = copy.deepcopy(base)
                wrapped = DDP(base, device_ids=[device.index], broadcast_buffers=False)
                n, batch = 37, 2
                batches = []
                for replica in range(parallel.dp_size):
                    g = torch.Generator().manual_seed(127+replica)
                    ids = torch.randint(31, (batch, n), generator=g)
                    target = torch.randint(31, (batch, n), generator=g)
                    mask = torch.ones_like(ids, dtype=torch.bool)
                    if kind == "tail":
                        mask[:, :-3] = False  # all but last CP rank have zero queries
                    elif kind == "ragged":
                        mask = torch.rand(batch, n, generator=g) > .65
                        mask[0] = False
                    batches.append((ids, target, mask))
                total_count = sum(int(b[2].sum()) for b in batches)
                reference_loss = torch.zeros((), device=device)
                for ids, target, mask in batches:
                    ref_out = ref(ids.to(device), targets=target.to(device), sup=mask.to(device))
                    reference_loss = reference_loss + ref_out["loss_sum"]/total_count
                reference_loss.backward()
                ids, target, mask = batches[parallel.dp_rank]
                ids, offset = parallel.shard(ids, base.block_size)
                target, _ = parallel.shard(target, base.block_size, -100)
                mask, _ = parallel.shard(mask, base.block_size, False)
                result = wrapped(ids.to(device), targets=target.to(device), sup=mask.to(device), parallel=parallel, offset=offset)
                global_count = sum_all(result["count"])
                self_loss = result["loss_sum"]*parallel.world_size/global_count
                self_loss.backward()
                torch.testing.assert_close(sum_all(result["loss_sum"])/global_count, reference_loss.detach(), rtol=2.e-6, atol=2.e-6)
                metrics = {}
                for (name, p), (_, rp) in zip(base.named_parameters(), ref.named_parameters()):
                    if rp.grad is not None:
                        torch.testing.assert_close(p.grad, rp.grad, rtol=5.e-5, atol=5.e-7, msg=name)
                        metrics[name] = error_metrics(p.grad, rp.grad)
                records.append({"operation": "distributed_gradient", "model": model_kind, "mask": kind, "cp": cp_size,
                                "dp": parallel.dp_size, "gradients": metrics})
        # Cross-rank global sparse top-k and causal halo warmup, including
        # weight-shared local/global passes and a non-divisible tail.
        torch.manual_seed(99)
        model = StackModel(31, 32, 8, 4, 2, "(L)x3,(G)x2", backend="torch", encoder_chunk=8).to(device).double().eval()
        generator = torch.Generator().manual_seed(83)
        ids = torch.randint(31, (2, 73), generator=generator)
        positions = torch.tensor([[0, 8, 31, 72], [1, 17, 50, 71]])
        sup = torch.zeros_like(ids, dtype=torch.bool).scatter_(1, positions, True)
        with torch.no_grad():
            expected = model(ids.to(device), sup=sup.to(device), compact=True)
            with InferenceSession(model, group=dist.group.WORLD, cache="cpu", page_tokens=12,
                                  encoder_chunk=8, query_chunk=2) as session:
                session.prefill(ids, positions)
                actual = torch.cat([out for _, out, _ in session.iter_logits()], 1)
                torch.testing.assert_close(actual, expected, atol=1.e-8, rtol=1.e-7)
        records.append({"operation": "distributed_sparse_prefill", **error_metrics(actual, expected)})
        return {"cases": records, "world": parallel.world_size}
    finally:
        dist.destroy_process_group()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["reference", "cuda", "distributed"], required=True)
    p.add_argument("--cp", type=int, default=2)
    p.add_argument("--output", default="validation.json")
    args = p.parse_args(argv)
    report = {"environment": environment(), "suite": args.suite, "passed": False}
    start = time.perf_counter()
    error = None
    try:
        report["results"] = (reference_suite() if args.suite == "reference" else
                             cuda_suite() if args.suite == "cuda" else distributed_suite(args.cp))
        report["passed"] = True
    except BaseException:
        error = traceback.format_exc()
        report["error"] = error
        if "st.kernels.launch" in sys.modules:
            from .kernels.launch import launch_report
            report["kernel_launches"] = launch_report()
    report["seconds"] = time.perf_counter()-start
    rank = int(os.environ.get("RANK", "0"))
    path = Path(args.output)
    if rank:
        path = path.with_name(path.stem+f".rank{rank}"+path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n")
    if error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"passed": True, "report": str(path)}), flush=True)


if __name__ == "__main__":
    main()
