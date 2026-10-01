"""Unified train/eval execution engine; invoked by st.cli and Python callers."""
import contextlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import checkpoint as ckpt
from .. import data
from .inference import InferenceSession
from .memory import GiB, training_estimate
from .parallel import ParallelContext, sum_all
from ..models.stack import StackModel
from ..models.baseline import BaselineModel
from ..api import build_model
from ..config import ModelConfig, ExecutionConfig
from ..data.tokens import TokenDataset, enwik8_dataset


def model_options(args):
    return ModelConfig(model=args.model, vocab_size=args.vocab_size, dim=args.dim,
                       heads=args.heads, block_size=args.block_size, arch=args.arch,
                       ffn_ratio=args.ffn_ratio, topk=args.topk or 64, layers=args.layers).to_dict()


def emit(record, args, rank=0):
    if rank:
        return
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")


def make_batch(args, generator, dataset=None):
    if args.task in ("tokens", "enwik8"):
        return dataset.batch(args.batch_size, args.length, generator)
    if args.task == "random":
        seq = torch.randint(args.vocab_size, (args.batch_size, args.length+1), generator=generator)
        return seq[:, :-1], seq[:, 1:], torch.ones(args.batch_size, args.length, dtype=torch.bool), None
    if args.task == "mqar":
        return data.mqar_batch(args.batch_size, args.length, generator, "cpu",
                               n_pairs=data.resolve_npairs(args.length, args.npairs, args.npairs_density),
                               n_queries=args.nqueries, key_tokens=args.nkeytoks)
    return getattr(data, f"{args.task}_batch")(args.batch_size, args.length, generator, "cpu")


def choose_cp(args, model, world, capacity):
    if args.context_parallel != "auto":
        cp = int(args.context_parallel)
        if cp < 1 or world % cp or model.heads % cp:
            raise ValueError("context_parallel must divide both WORLD_SIZE and heads")
        return cp
    candidates = [c for c in range(1, world+1) if world % c == 0 and model.heads % c == 0]
    for cp in candidates:
        estimate = training_estimate(model, args.batch_size, args.length, cp, args.checkpoint_chunks)
        if estimate["estimated_total_bytes"] < capacity*args.memory_fraction:
            return cp
    return candidates[-1]


def train(args, model, parallel, ck, dataset, validation_dataset=None):
    device = next(model.parameters()).device
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; choose fp32")
    if args.activation_offload and device.type != "cuda":
        raise ValueError("activation offload requires CUDA")
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False, gradient_as_bucket_view=True,
                  find_unused_parameters=isinstance(model, StackModel) and not bool(model.reads)) if parallel.world_size > 1 else model
    opt_kw = dict(lr=args.lr, betas=(.9, .95), weight_decay=args.weight_decay,
                  foreach=False)
    if args.optimizer_shard and parallel.world_size > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, **opt_kw)
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kw)
    generator = torch.Generator().manual_seed(args.seed+100+parallel.dp_rank)
    start = 0
    if ck is not None and not args.weights_only:
        topology = ck.get("topology", {})
        if topology and topology != {"world_size": parallel.world_size, "cp_size": parallel.cp_size}:
            raise ValueError("exact resume requires the saved topology; use --weights-only for a new topology")
        old_runtime = ck.get("config", {}).get("runtime", {})
        for key in ("task", "tokens", "token_dtype", "length", "batch_size", "grad_accum",
                    "precision", "steps", "lr", "weight_decay", "clip_grad", "npairs", "npairs_density", "nqueries", "nkeytoks"):
            if key in old_runtime and getattr(args, key) != old_runtime[key]:
                raise ValueError(f"exact resume requires {key}={old_runtime[key]!r}; use --weights-only to change the experiment")
        opt.load_state_dict(ck["opt"])
        start = ck["step"]+1
        if "rng_by_rank" in ck:
            ckpt.restore_rng(ck["rng_by_rank"][parallel.rank], generator)
        else:
            emit({"event": "warning",
                  "message": "legacy checkpoint has no RNG state; data stream restarts"}, args, parallel.rank)
    if ck is not None:
        ck.clear()  # do not retain a full CPU optimizer copy on every rank
    model_keys = set(ModelConfig.__dataclass_fields__)
    config = {"model": model_options(args), "runtime": {k: v for k, v in vars(args).items() if k not in model_keys and k not in ("command", "config")}}
    if start >= args.steps:
        raise ValueError("--steps is the total target step count and must exceed the resumed step")
    model.train()
    last_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stats = torch.zeros(3, device=device, dtype=torch.float64)
    window_steps = 0
    for step in range(start, args.steps):
        opt.zero_grad(set_to_none=True)
        warm = max(1, args.steps//20)
        schedule = ((step+1)/warm if step < warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,args.steps-warm))))
        for group in opt.param_groups:
            group["lr"] = args.lr*schedule
        for micro in range(args.grad_accum):
            ids, target, mask, _ = make_batch(args, generator, dataset)
            # Each CP group generates the same CPU batch; data replicas use
            # independent deterministic streams. Only local shards hit GPU.
            ids, offset = parallel.shard(ids, model.block_size)
            target, _ = parallel.shard(target, model.block_size, -100)
            mask, _ = parallel.shard(mask, model.block_size, False)
            ids, target, mask = (t.to(device, non_blocking=True) for t in (ids, target, mask))
            sync = wrapped.no_sync() if isinstance(wrapped, DDP) and micro < args.grad_accum-1 else contextlib.nullcontext()
            amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16")
            offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.activation_offload else contextlib.nullcontext()
            with sync:
                with amp, offload:
                    result = wrapped(ids, sup=mask, targets=target, parallel=parallel, offset=offset)
                    count = sum_all(result["count"])
                    if int(count) == 0:
                        raise ValueError("global batch has no supervised tokens")
                    scaled = result["loss_sum"]*parallel.world_size/count/args.grad_accum
                if not bool(torch.isfinite(sum_all(scaled.detach()))):
                    raise FloatingPointError("non-finite loss across ranks; optimizer step was not applied")
                scaled.backward()
            stats += torch.stack((result["loss_sum"].detach().double(), result["count"].double(), result["correct"].double()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
        opt.step()
        window_steps += 1
        if (step+1) % args.log_every == 0 or step == args.steps-1:
            totals = sum_all(stats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-last_time
            timing = torch.tensor(elapsed, device=device)
            if parallel.world_size > 1:
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
            emit({"event": "train", "step": step+1, "loss": round(float(totals[0]/totals[1]), 6),
                  "accuracy": round(float(totals[2]/totals[1]), 4), "grad_norm": round(float(norm), 4),
                  "tokens_per_second": round(window_steps*args.grad_accum*args.batch_size*parallel.dp_size*args.length/float(timing)),
                  "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device)/GiB, 3) if device.type == "cuda" else None,
                  "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device)/GiB, 3) if device.type == "cuda" else None}, args, parallel.rank)
            stats.zero_()
            last_time, window_steps = time.perf_counter(), 0
        should_stop = False
        if args.eval_every and ((step+1) % args.eval_every == 0 or step == args.steps-1):
            evaluation_start = time.perf_counter()
            rec = evaluate(args, model, parallel, validation_dataset, during_training=True, step=step+1)
            model.train()
            should_stop = args.stop_exact is not None and rec["exact"] >= args.stop_exact
            last_time += time.perf_counter()-evaluation_start
        if args.save and (should_stop or step == args.steps-1 or args.save_every and (step+1) % args.save_every == 0):
            ckpt.save(args.save, model, opt, config, step, generator, parallel)
        if should_stop:
            emit({"event": "early_stop", "step": step+1, "exact": round(rec["exact"], 4)}, args, parallel.rank)
            break


@torch.no_grad()
def evaluate(args, model, parallel, dataset, during_training=False, step=None):
    model.eval()
    if args.precision == "bf16" and not during_training:
        model.to(torch.bfloat16)
    group = dist.group.WORLD if parallel.world_size > 1 else None
    generator = torch.Generator().manual_seed(args.seed+200)
    loss_sum = count = correct = exact = rows_total = 0
    begin = time.perf_counter()
    if next(model.parameters()).is_cuda and not during_training:
        torch.cuda.reset_peak_memory_stats()
    for batch_index in range(args.eval_batches):
        ids, targets, mask, _ = make_batch(args, generator, dataset)
        if args.eval_positions:
            width = min(args.eval_positions, args.length)
            # Deterministic stratified coverage, including the last token.
            positions = torch.linspace(0, args.length-1, width).round().long()[None].expand(args.batch_size, -1)
        else:
            counts = mask.sum(1)
            if int(counts.max()) != int(counts.min()) or int(counts.min()) == 0:
                raise ValueError("inference task masks must have equal nonzero counts per row")
            positions = mask.nonzero()[:, 1].reshape(args.batch_size, -1)
        truth = targets.gather(1, positions)
        errors = torch.zeros(args.batch_size, dtype=torch.long)
        if isinstance(model, StackModel):
            session_context = InferenceSession(model, group=group, cache=args.cache, cache_dir=args.cache_dir or None,
                    memory_fraction=args.memory_fraction, page_tokens=args.page_tokens,
                    encoder_chunk=args.encoder_chunk, query_chunk=args.inference_query_chunk,
                    workspace_mb=args.workspace_mb, max_query_states_mb=args.max_query_states_mb)
        else:
            session_context = contextlib.nullcontext()
        with session_context as session:
            if session is not None:
                session.prefill(ids, positions)
                if batch_index == 0:
                    emit({"event": "inference_plan", "batch": batch_index, **session.plan.to_dict()}, args, parallel.rank)
                iterator = session.iter_logits()
            else:
                device = next(model.parameters()).device
                iterator = model.iter_logits(ids.to(device), positions.to(device))
            for sl, logits, _ in iterator:
                target = truth[:, sl].to(logits.device)
                loss_sum += float(torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), target.flatten(), reduction="sum"))
                hit = logits.argmax(-1).eq(target)
                count += target.numel()
                correct += int(hit.sum())
                errors += (~hit).sum(1).cpu()
        exact += int((errors == 0).sum())
        rows_total += args.batch_size
    result = {"event": "eval", **({"step": step} if step is not None else {}),
              "length": args.length, "loss": round(loss_sum/count, 6),
              "bpc": round(loss_sum/count/math.log(2), 6), "accuracy": round(correct/count, 4),
              "exact": round(exact/rows_total, 4), "evaluated_tokens": count,
              "seconds": round(time.perf_counter()-begin, 3),
              "peak_allocated_gib": round(torch.cuda.max_memory_allocated()/GiB, 3) if next(model.parameters()).is_cuda else None}
    emit(result, args, parallel.rank)
    if args.output and parallel.rank == 0 and not during_training:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    return result


def run(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ck = ckpt.load(args.resume) if args.resume else None
    if args.task == "enwik8":
        args.vocab_size = 256
    options = dict(ckpt.model_config(ck)) if ck is not None else model_options(args)
    options.setdefault("model", "stack")
    if args.topk is not None and options["model"] == "stack":
        options["topk"] = args.topk
    for key, value in options.items():
        setattr(args, key, value)
    args.topk = options.get("topk", 64)
    model = build_model(options, device="meta" if args.command == "plan" else "cpu",
                        execution=ExecutionConfig(args.backend, args.checkpoint_chunks,
                                                  args.encoder_chunk, args.query_chunk, args.loss_chunk))
    if ck is not None:
        if args.command != "plan":
            model.load_state_dict(ck["model"], strict=True)
        ck.pop("model")
        if args.command != "train":
            ck.clear()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not 1 <= world <= 8:
        raise ValueError("supported process count is 1..8")
    if args.command == "plan":
        estimates = [training_estimate(model, args.batch_size, args.length, c, args.checkpoint_chunks)
                     for c in (1, 2, 4, 8) if model.heads % c == 0]
        print(json.dumps({"model": options, "training": estimates,
                          "inference_raw_kv_bytes": 2*args.batch_size*args.length*model.dim*(2 if args.precision == "bf16" else 4) if isinstance(model, StackModel) else None}, indent=2))
        return
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA is required; use --device cpu only for reference validation")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda", torch.cuda.current_device())
        capacity = torch.cuda.mem_get_info(device)[1]
    else:
        device, capacity = torch.device("cpu"), 64*GiB
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    if world > 1:
        cap = torch.tensor(capacity, device=device, dtype=torch.long)
        dist.all_reduce(cap, op=dist.ReduceOp.MIN)
        capacity = int(cap)
    cp = choose_cp(args, model, world, capacity) if args.command == "train" else 1
    parallel = ParallelContext.initialize(cp, args.device)
    try:
        estimate = training_estimate(model, args.batch_size, args.length, cp, args.checkpoint_chunks)
        estimate.pop("note")
        hint = None
        if world == 1 and args.device == "cuda" and torch.cuda.device_count() > 1:
            hint = (f"{torch.cuda.device_count()} GPUs are visible; "
                    "launch with torchrun --nproc_per_node=N to train on all of them")
        emit({"event": "configuration", "model": options, "world": world, "cp": cp,
              "dp": parallel.dp_size, "effective_batch": args.batch_size*parallel.dp_size*args.grad_accum,
              **estimate, **({"hint": hint} if hint else {})}, args, parallel.rank)
        if args.command == "train" and device.type == "cuda" and estimate["estimated_total_bytes"] > capacity*args.memory_fraction and not args.allow_over_budget:
            raise MemoryError("training estimate exceeds memory budget; reduce microbatch/use CP/checkpointing, or inspect --allow-over-budget")
        if args.task in ("passkey", "copying", "mqar") and args.vocab_size < data.VOCAB:
            raise ValueError("synthetic tasks require vocab_size >= 128")
        validation_dataset = None
        if args.task == "tokens":
            dataset = TokenDataset(args.tokens, args.token_dtype)
            if args.validation_tokens:
                validation_dataset = TokenDataset(args.validation_tokens, args.token_dtype)
        elif args.task == "enwik8":
            dataset = enwik8_dataset(args.tokens or "data/enwik8", "train" if args.command == "train" else args.split)
            if args.command == "train" and args.eval_every:
                validation_dataset = enwik8_dataset(args.tokens or "data/enwik8", args.split)
        else:
            dataset = None
        model.to(device)
        if args.command == "train":
            return train(args, model, parallel, ck, dataset, validation_dataset)
        else:
            return evaluate(args, model, parallel, dataset)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
