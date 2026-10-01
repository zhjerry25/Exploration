"""Conservative capacity planning; estimates are not a promise of throughput."""
from dataclasses import asdict, dataclass
import os
import shutil

import torch


GiB = 1 << 30


def available_host_memory():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])*1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None


@dataclass
class InferencePlan:
    cache: str
    cache_bytes: int
    gpu_budget_bytes: int
    page_tokens: int
    encoder_chunk: int
    query_chunk: int
    score_tokens: int
    head_chunk: int
    host_available_bytes: int | None

    def to_dict(self):
        return asdict(self)


def plan_inference(model, batch, local_length, *, cache="auto", memory_fraction=.8,
                   page_tokens=65536, encoder_chunk=4096, query_chunk=16,
                   workspace_mb=256, cache_dir=None):
    if not 0 < memory_fraction < 1 or min(batch, local_length, page_tokens, encoder_chunk, query_chunk, workspace_mb) < 1:
        raise ValueError("positive sizes and 0 < memory_fraction < 1 required")
    if cache not in ("auto", "cuda", "cpu", "disk"):
        raise ValueError("cache must be auto/cuda/cpu/disk")
    param = next(model.parameters())
    b, dim, heads = model.block_size, model.dim, model.heads
    element = param.element_size()
    total = 2*batch*local_length*dim*element
    if param.is_cuda:
        free, capacity = torch.cuda.mem_get_info(param.device)
        # mem_get_info is free AFTER model allocation. Keep explicit slack
        # for GEMM/SDPA/NCCL workspaces and allocator fragmentation.
        budget = max(0, min(int(capacity*memory_fraction), free-(2*GiB)))
    else:
        budget = 0
    workspace = workspace_mb*(1 << 20)
    host = available_host_memory()
    # Host RAM/disk are shared by local ranks, GPU capacity is per rank.
    local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    host_budget = None if host is None else host/max(1, local_processes)
    if cache == "auto":
        if param.is_cuda and total + 4*workspace < budget:
            cache = "cuda"
        elif host_budget is not None and total + 4*workspace < host_budget*.7:
            cache = "cpu"
        elif host_budget is None and total + 4*workspace <= 2*GiB:
            # Host RAM is undeterminable (e.g. macOS without psutil) and the
            # cache is trivially small; proceed on CPU rather than refuse.
            cache = "cpu"
        elif cache_dir:
            cache = "disk"
        else:
            raise MemoryError(f"KV requires {total/GiB:.2f} GiB per rank; supply --cache-dir for disk paging or more inference ranks")
    if cache == "cuda" and (not param.is_cuda or total+4*workspace >= budget):
        raise MemoryError(f"GPU KV requires {total/GiB:.2f} GiB plus workspace; select --cache auto/cpu/disk")
    if cache == "cpu" and host_budget is not None and total+4*workspace > host_budget*.7:
        raise MemoryError(f"CPU KV requires {total/GiB:.2f} GiB; available RAM {host/GiB:.2f} GiB; select disk paging")
    if cache == "disk":
        if not cache_dir:
            raise ValueError("disk cache requires --cache-dir on a filesystem with sufficient free space")
        os.makedirs(cache_dir, exist_ok=True)
        if (total + GiB)*local_processes > shutil.disk_usage(cache_dir).free:
            raise MemoryError("insufficient free disk space for the local KV shard")
    # Storage and transfer pages are independently bounded from score tiles.
    page = max(b, min(page_tokens, workspace//max(2*batch*dim*element, 1))//b*b)
    enc = max(b, min(encoder_chunk, workspace//max(24*batch*dim*element, 1))//b*b)
    qc = max(1, min(query_chunk, workspace//max(batch*heads*model.topk*b*4, 1)))
    score = max(b, min(page, workspace//max(4*batch*heads*qc*4, 1))//b*b)
    head_chunk = max(1, min(qc, workspace//max(batch*model.lm_head.out_features*4, 1)))
    return InferencePlan(cache, total, budget, page, enc, qc, score, head_chunk, host)


def training_estimate(model, batch, length, cp_size=1, checkpoint_chunks=True):
    params = sum(p.numel() for p in model.parameters())
    local = ((length+cp_size*model.block_size-1)//(cp_size*model.block_size))*model.block_size
    # fp32 weights/grad/m/v. Conservative activation estimate, includes
    # multiple global-read passes, local residuals and all-to-all buffers.
    param_bytes = params*16
    layers = len(model.local) if hasattr(model, "local") else len(model.blocks)
    reads = len(model.reads) if hasattr(model, "reads") else layers
    factor = (layers*2+reads*8+12) if checkpoint_chunks else (layers*16+reads*20+12)
    activations = batch*local*model.dim*4*factor
    return {"parameters": params, "parameter_optimizer_bytes": param_bytes,
            "estimated_activation_bytes": activations,
            "estimated_total_bytes": param_bytes+activations+2*GiB,
            "local_tokens": local, "context_parallel": cp_size,
            "note": "Capacity estimate; remote peak-memory measurement is required."}
