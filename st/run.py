"""Scalable experiment CLI: python -m st.run {plan,train,eval} --help."""
import argparse
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
from . import data
from .inference import InferenceSession
from .memory import GiB, training_estimate
from .parallel import ParallelContext, sum_all
from .stack_model import StackModel
from .token_data import TokenDataset


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "train", "eval"])
    p.add_argument("--config", help="JSON with model and runtime objects")
    p.add_argument("--task", choices=["passkey", "mqar", "copying", "tokens", "random"], default="passkey")
    p.add_argument("--tokens", default="", help="flat token corpus (required for task=tokens)")
    p.add_argument("--token-dtype", default="uint16", choices=["uint8", "uint16", "int32", "int64"])
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--arch", default="Lx2,G")
    p.add_argument("--ffn-ratio", type=float, default=4.)
    p.add_argument("--topk", type=int, default=None, help="inference top-k; training is always dense")
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1, help="microbatch per data-parallel replica")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--context-parallel", default="auto", help="auto or a divisor of heads and WORLD_SIZE")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--backend", choices=["auto", "torch", "triton"], default="auto")
    p.add_argument("--checkpoint-chunks", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--activation-offload", action="store_true", help="offload autograd saved tensors to pinned host RAM")
    p.add_argument("--optimizer-shard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--encoder-chunk", type=int, default=1024)
    p.add_argument("--query-chunk", type=int, default=128, help="local training queries per chunk")
    p.add_argument("--loss-chunk", type=int, default=128)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3.e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--clip-grad", type=float, default=1.)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--save", default="runs/stack.pt")
    p.add_argument("--resume", default="")
    p.add_argument("--weights-only", action="store_true", help="load weights, reset optimizer and RNG")
    p.add_argument("--log", default="", help="rank-zero JSONL path")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--memory-fraction", type=float, default=.8)
    p.add_argument("--allow-over-budget", action="store_true", help="override a conservative training estimate; never changes semantics")
    p.add_argument("--npairs", type=int, default=16)
    p.add_argument("--nqueries", type=int, default=4)
    p.add_argument("--nkeytoks", type=int, choices=[1, 2], default=1)
    p.add_argument("--eval-batches", type=int, default=1)
    p.add_argument("--eval-positions", type=int, default=0, help="0=task mask; otherwise uniformly sampled query positions")
    p.add_argument("--cache", default="auto", choices=["auto", "cuda", "cpu", "disk"])
    p.add_argument("--cache-dir", default="")
    p.add_argument("--page-tokens", type=int, default=65536)
    p.add_argument("--inference-query-chunk", type=int, default=16)
    p.add_argument("--workspace-mb", type=int, default=256)
    p.add_argument("--max-query-states-mb", type=int, default=256)
    p.add_argument("--output", default="", help="eval JSON output, written by rank zero")
    return p


def parse_args(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        with open(preliminary.config) as f:
            config = json.load(f)
        if set(config)-{"model", "runtime"}:
            raise ValueError("config root accepts only model and runtime")
        defaults = dict(config.get("model", {}), **config.get("runtime", {}))
        known = {a.dest for a in p._actions}
        if set(defaults)-known:
            raise ValueError(f"unknown config keys: {sorted(set(defaults)-known)}")
        p.set_defaults(**defaults)
    args = p.parse_args(argv)
    positive = ("length", "batch_size", "grad_accum", "encoder_chunk", "query_chunk", "loss_chunk",
                "steps", "log_every", "eval_batches", "page_tokens", "inference_query_chunk", "workspace_mb",
                "max_query_states_mb")
    for name in positive:
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if not 0 < args.memory_fraction < 1:
        p.error("memory_fraction must lie strictly between 0 and 1")
    if args.command == "train" and args.length > 65536:
        p.error("dense training is limited to 65536 tokens; larger lengths are inference-only")
    if args.command == "eval" and not args.resume:
        p.error("eval requires --resume")
    if args.weights_only and not args.resume:
        p.error("weights-only requires --resume")
    return args


def model_options(args):
    return {"vocab_size": args.vocab_size, "dim": args.dim, "heads": args.heads,
            "block_size": args.block_size, "arch": args.arch, "ffn_ratio": args.ffn_ratio,
            "topk": args.topk or 64}


def emit(record, args, rank=0):
    if rank:
        return
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")


def make_batch(args, generator, dataset=None):
    if args.task == "tokens":
        return dataset.batch(args.batch_size, args.length, generator)
    if args.task == "random":
        seq = torch.randint(args.vocab_size, (args.batch_size, args.length+1), generator=generator)
        return seq[:, :-1], seq[:, 1:], torch.ones(args.batch_size, args.length, dtype=torch.bool), None
    if args.task == "mqar":
        return data.mqar_batch(args.batch_size, args.length, generator, "cpu",
                               n_pairs=args.npairs, n_queries=args.nqueries, key_tokens=args.nkeytoks)
    return getattr(data, f"{args.task}_batch")(args.batch_size, args.length, generator, "cpu")


def choose_cp(args, model, world, capacity):
    if args.context_parallel != "auto":
        cp = int(args.context_parallel)
        if cp < 1 or world % cp or model.heads % cp:
            raise ValueError("context_parallel must divide both WORLD_SIZE and heads")
        return cp
    candidates = [c for c in range(1, world+1) if world % c == 0 and model.heads % c == 0]
    # Full-token losses at long n benefit from parallelizing the dense read.
    if args.length >= 8192 and args.task in ("tokens", "random", "copying"):
        return candidates[-1]
    for cp in candidates:
        estimate = training_estimate(model, args.batch_size, args.length, cp, args.checkpoint_chunks)
        if estimate["estimated_total_bytes"] < capacity*args.memory_fraction:
            return cp
    return candidates[-1]


def train(args, model, parallel, ck, dataset):
    device = next(model.parameters()).device
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; choose fp32")
    if args.activation_offload and device.type != "cuda":
        raise ValueError("activation offload requires CUDA")
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False, gradient_as_bucket_view=True,
                  find_unused_parameters=not bool(model.reads)) if parallel.world_size > 1 else model
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
                    "precision", "steps", "lr", "weight_decay", "clip_grad", "npairs", "nqueries", "nkeytoks"):
            if key in old_runtime and getattr(args, key) != old_runtime[key]:
                raise ValueError(f"exact resume requires {key}={old_runtime[key]!r}; use --weights-only to change the experiment")
        opt.load_state_dict(ck["opt"])
        start = ck["step"]+1
        if "rng_by_rank" in ck:
            ckpt.restore_rng(ck["rng_by_rank"][parallel.rank], generator)
        else:
            emit({"warning": "legacy checkpoint has no RNG state; data stream restarts"}, args, parallel.rank)
    if ck is not None:
        ck.clear()  # do not retain a full CPU optimizer copy on every rank
    config = {"model": model_options(args), "runtime": {k: v for k, v in vars(args).items() if k not in model_options(args)}}
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
            emit({"event": "train", "step": step+1, "loss": float(totals[0]/totals[1]),
                  "accuracy": float(totals[2]/totals[1]), "grad_norm": float(norm),
                  "tokens_per_second": window_steps*args.grad_accum*args.batch_size*parallel.dp_size*args.length/float(timing),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/GiB if device.type == "cuda" else None,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/GiB if device.type == "cuda" else None}, args, parallel.rank)
            stats.zero_()
            last_time, window_steps = time.perf_counter(), 0
        if args.save and (step == args.steps-1 or args.save_every and (step+1) % args.save_every == 0):
            ckpt.save(args.save, model, opt, config, step, generator, parallel)


@torch.no_grad()
def evaluate(args, model, parallel, dataset):
    model.eval()
    if args.precision == "bf16":
        model.to(torch.bfloat16)
    group = dist.group.WORLD if parallel.world_size > 1 else None
    generator = torch.Generator().manual_seed(args.seed+200)
    loss_sum = count = correct = exact = rows_total = 0
    begin = time.perf_counter()
    if next(model.parameters()).is_cuda:
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
        with InferenceSession(model, group=group, cache=args.cache, cache_dir=args.cache_dir or None,
                memory_fraction=args.memory_fraction, page_tokens=args.page_tokens,
                encoder_chunk=args.encoder_chunk, query_chunk=args.inference_query_chunk,
                workspace_mb=args.workspace_mb, max_query_states_mb=args.max_query_states_mb) as session:
            session.prefill(ids, positions)
            emit({"event": "inference_plan", "batch": batch_index, **session.plan.to_dict()}, args, parallel.rank)
            for sl, logits, _ in session.iter_logits():
                target = truth[:, sl].to(logits.device)
                loss_sum += float(torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), target.flatten(), reduction="sum"))
                hit = logits.argmax(-1).eq(target)
                count += target.numel()
                correct += int(hit.sum())
                errors += (~hit).sum(1).cpu()
        exact += int((errors == 0).sum())
        rows_total += args.batch_size
    result = {"event": "eval", "length": args.length, "loss": loss_sum/count,
              "bpc": loss_sum/count/math.log(2), "accuracy": correct/count,
              "exact": exact/rows_total, "evaluated_tokens": count,
              "seconds": time.perf_counter()-begin,
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/GiB if next(model.parameters()).is_cuda else None}
    emit(result, args, parallel.rank)
    if args.output and parallel.rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ck = ckpt.load(args.resume) if args.resume else None
    options = ckpt.model_config(ck) if ck is not None else model_options(args)
    if args.topk is not None:
        options["topk"] = args.topk
    for key, value in options.items():
        setattr(args, key, value)
    args.topk = options.get("topk", 64)
    with torch.device("meta" if args.command == "plan" else "cpu"):
        model = StackModel(**options, backend=args.backend, pos_chunk=args.query_chunk,
                           checkpoint_chunks=args.checkpoint_chunks,
                           encoder_chunk=args.encoder_chunk, loss_chunk=args.loss_chunk)
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
                          "inference_kv_bytes": 2*args.batch_size*args.length*model.dim*(2 if args.precision == "bf16" else 4)}, indent=2))
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
        emit({"event": "configuration", "model": options, "world": world, "cp": cp,
              "dp": parallel.dp_size, "effective_batch": args.batch_size*parallel.dp_size*args.grad_accum,
              **estimate}, args, parallel.rank)
        if args.command == "train" and device.type == "cuda" and estimate["estimated_total_bytes"] > capacity*args.memory_fraction and not args.allow_over_budget:
            raise MemoryError("training estimate exceeds memory budget; reduce microbatch/use CP/checkpointing, or inspect --allow-over-budget")
        if args.task in ("passkey", "copying", "mqar") and args.vocab_size < data.VOCAB:
            raise ValueError("synthetic tasks require vocab_size >= 128")
        dataset = TokenDataset(args.tokens, args.token_dtype) if args.task == "tokens" else None
        model.to(device)
        if args.command == "train":
            train(args, model, parallel, ck, dataset)
        else:
            evaluate(args, model, parallel, dataset)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
