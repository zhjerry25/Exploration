"""Resource-aware Triton launches, with no fallback to a different algorithm.

Only OutOfResources is retried: Triton raises it before kernel execution.
Compilation errors, device OOMs and numerical failures remain fatal.
"""
import torch
from triton.runtime.errors import OutOfResources


_chosen = {}
_records = {}


def launch(kernel, grid, args, meta, device, dtype):
    name = kernel.__name__
    device = torch.device(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (name, device_index, str(dtype), tuple(sorted(meta.items())))
    if key in _chosen:
        candidate = _chosen[key]
        compiled = kernel[grid(candidate)](*args, **candidate)
        return candidate
    # IEEE fp32 uses more shared memory than bf16 tensor-core paths. In
    # particular D=96 rounds to DM=128; default three-stage pipelining exceeded
    # the reported Pro 6000 per-block limit (143424 > 101376 bytes).
    first_stages = 1 if dtype == torch.float32 else 2
    ns = [meta["N"]]
    if "B" in meta:
        while ns[-1] > max(16, meta["B"]):
            ns.append(ns[-1]//2)
    candidates = []
    for n in ns:
        for stages in dict.fromkeys((first_stages, 1)):
            for warps in (4, 8, 2):
                candidates.append(dict(meta, N=n, num_stages=stages, num_warps=warps))
    last_error = None
    for candidate in candidates:
        try:
            compiled = kernel[grid(candidate)](*args, **candidate)
        except OutOfResources as exc:
            last_error = exc
            continue
        _chosen[key] = candidate
        _records[key] = {"kernel": name, "device": device_index, "dtype": str(dtype),
                         "head_dim": meta.get("D"), "block_size": meta.get("B"),
                         "tile_m": candidate.get("M"), "tile_n": candidate["N"],
                         "num_stages": candidate["num_stages"], "num_warps": candidate["num_warps"],
                         "shared_memory_bytes": getattr(compiled.metadata, "shared", None)}
        return candidate
    raise RuntimeError(f"No resource-compatible launch for {name}, D={meta.get('D')}, "
                       f"block_size={meta.get('B')}, dtype={dtype}; last failure: {last_error}") from last_error


def launch_report():
    """Compact records of successful configurations for validation reports."""
    return list(_records.values())
