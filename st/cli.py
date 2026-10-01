"""Single CLI: python -m st {plan,train,eval,validate,benchmark}."""
import argparse
import json
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "train", "eval", "validate", "benchmark"])
    p.add_argument("--config", help="JSON with model and runtime objects")
    p.add_argument("--model", choices=["stack", "baseline"], default="stack")
    p.add_argument("--layers", type=int, default=3, help="baseline depth")
    p.add_argument("--task", choices=["passkey", "mqar", "copying", "tokens", "enwik8", "random"], default="passkey")
    p.add_argument("--tokens", default="", help="flat token corpus (required for task=tokens)")
    p.add_argument("--token-dtype", default="uint16", choices=["uint8", "uint16", "int32", "int64"])
    p.add_argument("--validation-tokens", default="", help="separate validation token file for periodic evaluation")
    p.add_argument("--split", choices=["train", "val", "test"], default="val", help="enwik8 evaluation split")
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
    p.add_argument("--eval-every", type=int, default=0, help="periodic evaluation interval; 0 disables it")
    p.add_argument("--stop-exact", type=float, default=None, help="save and stop when periodic exact accuracy reaches threshold")
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
    p.add_argument("--npairs-density", type=float, default=0., help="MQAR pairs per token, overriding fixed npairs")
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
        if not isinstance(config, dict) or any(not isinstance(config.get(k, {}), dict) for k in ("model", "runtime")):
            p.error("config must contain model/runtime JSON objects")
        if set(config)-{"model", "runtime"}:
            raise ValueError("config root accepts only model and runtime")
        defaults = dict(config.get("model", {}), **config.get("runtime", {}))
        known = {a.dest for a in p._actions} - {"command", "config", "help"}
        if set(defaults)-known:
            raise ValueError(f"unknown config keys: {sorted(set(defaults)-known)}")
        p.set_defaults(**defaults)
    args = p.parse_args(argv)
    # argparse does not validate choices or non-string defaults supplied by JSON.
    for action in p._actions:
        if action.dest == "help":
            continue
        value = getattr(args, action.dest, None)
        if value is None:
            continue
        if action.type is int and type(value) is not int:
            p.error(f"{action.dest} must be an integer")
        if action.type is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
            p.error(f"{action.dest} must be a number")
        if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)) and type(value) is not bool:
            p.error(f"{action.dest} must be a boolean")
        if action.choices is not None and value not in action.choices:
            p.error(f"invalid {action.dest}: {value!r}; choose from {action.choices}")
    positive = ("length", "batch_size", "grad_accum", "encoder_chunk", "query_chunk", "loss_chunk",
                "steps", "log_every", "eval_batches", "page_tokens", "inference_query_chunk", "workspace_mb",
                "max_query_states_mb")
    for name in positive:
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if not 0 < args.memory_fraction < 1:
        p.error("memory_fraction must lie strictly between 0 and 1")
    if args.context_parallel != "auto":
        try:
            if not str(args.context_parallel).isdigit() or int(args.context_parallel) < 1:
                raise ValueError
            args.context_parallel = str(int(args.context_parallel))
        except (TypeError, ValueError):
            p.error("context_parallel must be auto or a positive integer")
    if args.command == "train" and args.length > 65536:
        p.error("dense training is limited to 65536 tokens; larger lengths are inference-only")
    if args.command == "eval" and not args.resume:
        p.error("eval requires --resume")
    if args.weights_only and not args.resume:
        p.error("weights-only requires --resume")
    if args.topk is not None and args.topk < 1:
        p.error("topk must be positive")
    if min(args.eval_every, args.save_every, args.eval_positions, args.npairs_density) < 0:
        p.error("intervals, eval_positions and npairs_density must be nonnegative")
    if args.stop_exact is not None and (not 0 <= args.stop_exact <= 1 or not args.eval_every):
        p.error("stop-exact requires a threshold in [0,1] and eval-every > 0")
    if args.command == "train" and args.task == "tokens" and args.eval_every and not args.validation_tokens:
        p.error("periodic token evaluation requires --validation-tokens (a held-out corpus)")
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "validate":
        from .validate import main as validate
        return validate(argv[1:])
    if argv and argv[0] == "benchmark":
        from .benchmark import main as benchmark
        return benchmark(argv[1:])
    from .engine import run
    return run(parse_args(argv))


def entrypoint():
    """Console scripts must not use a result dictionary as an exit status."""
    main()
