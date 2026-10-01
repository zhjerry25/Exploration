"""Portable atomic checkpoints, including per-rank data/RNG state."""
import os
from pathlib import Path
import random
import tempfile

import torch
import torch.distributed as dist


def rng_state(generator):
    state = {"torch": torch.get_rng_state(), "python": random.getstate(),
             "data": generator.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    generator.set_state(state["data"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def _cpu_model_state(model):
    # Preserve aliases for tied embedding/read layers. Copying each state
    # dictionary key separately can multiply shared-weight checkpoints.
    memo, result = {}, {}
    for name, value in model.state_dict().items():
        key = (value.device, value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
        if key not in memo:
            memo[key] = value.detach().cpu()
        result[name] = memo[key]
    return result


def save(path, model, optimizer, config, step, generator, parallel):
    if hasattr(optimizer, "consolidate_state_dict"):
        optimizer.consolidate_state_dict(to=0)
    local = rng_state(generator)
    if parallel.world_size > 1:
        states = [None]*parallel.world_size if parallel.rank == 0 else None
        dist.gather_object(local, states, dst=0)
    else:
        states = [local]
    error = None
    if parallel.rank == 0:
        temp_path = None
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            state = {"format_version": 2, "model": _cpu_model_state(model),
                     "opt": optimizer.state_dict(), "config": config, "step": step,
                     "rng_by_rank": states,
                     "topology": {"world_size": parallel.world_size, "cp_size": parallel.cp_size}}
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+".", suffix=".tmp", delete=False) as f:
                temp_path = f.name
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target)
        except Exception as exc:
            error = f"checkpoint save failed: {exc}"
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    if parallel.world_size > 1:
        message = [error]
        dist.broadcast_object_list(message, src=0)
        error = message[0]
    if error:
        raise RuntimeError(error)


def load(path):
    # Framework checkpoints contain optimizer and Python RNG state. Only
    # load locally trusted checkpoints (same contract as the legacy driver).
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def model_config(ck):
    if "config" in ck:
        return ck["config"]["model"]
    args = ck.get("args", {})
    if args.get("model", "stack") == "baseline":
        return dict(model="baseline", vocab_size=256 if args.get("task") == "lm" else 128,
                    dim=args.get("d", 256), heads=args.get("heads", 4),
                    layers=args.get("layers", 3), ffn_ratio=args.get("ffn_ratio", 4))
    return dict(vocab_size=256 if args.get("task") == "lm" else 128,
                dim=args.get("d", 256), heads=args.get("heads", 4),
                block_size=args.get("b", 16), topk=args.get("read_m") or 64,
                arch=args.get("arch") or "Lx2,G", ffn_ratio=args.get("ffn_ratio", 4))
