"""Single CLI: python -m st {plan,train,eval,validate,benchmark}."""
import argparse
import json
import sys


COMMANDS = {
    "plan": "Estimate model size and memory without allocating weights",
    "train": "Train with dense attention; optionally evaluate and checkpoint",
    "eval": "Evaluate a checkpoint; StackModel uses paged sparse inference",
    "validate": "Run reference, CUDA parity or multi-GPU correctness checks",
    "benchmark": "Measure attention latency/memory on a CUDA GPU",
}


def _experiment_options(p, command):
    model = {"model", "layers", "vocab_size", "dim", "heads", "block_size", "arch", "ffn_ratio", "topk"}
    data = {"task", "tokens", "token_dtype", "validation_tokens", "split", "length", "batch_size", "npairs", "npairs_density", "nqueries", "nkeytoks", "seed"}
    train = {"grad_accum", "steps", "eval_every", "stop_exact", "lr", "weight_decay", "clip_grad", "log_every", "save_every", "save", "weights_only", "optimizer_shard", "activation_offload", "allow_over_budget", "context_parallel"}
    evaluation = {"eval_batches", "eval_positions", "cache", "cache_dir", "page_tokens", "inference_query_chunk", "workspace_mb", "max_query_states_mb"}
    groups = {name: p.add_argument_group(name) for name in ("model", "data", "execution", "training", "evaluation", "files")}
    plan_fields = model | {"config", "task", "length", "batch_size", "precision", "checkpoint_chunks", "resume", "encoder_chunk", "query_chunk", "loss_chunk", "backend"}

    def add(flag, **options):
        dest = flag[2:].replace("-", "_")
        visible = (command != "plan" or dest in plan_fields) and (command != "eval" or dest not in train)
        if not visible:
            p.set_defaults(**{dest: options.get("default", False if options.get("action") == "store_true" else None)})
            return
        group = ("model" if dest in model else "data" if dest in data else
                 "training" if dest in train else "evaluation" if dest in evaluation else
                 "files" if dest in {"config", "resume", "output", "log"} else "execution")
        groups[group].add_argument(flag, **options)
    add("--config", help="JSON with model and runtime objects")
    add("--model", choices=["stack", "baseline"], default="stack")
    add("--layers", type=int, default=3, help="baseline depth")
    add("--task", choices=["passkey", "mqar", "copying", "tokens", "enwik8", "random"], default="passkey")
    add("--tokens", default="", help="flat token corpus (required for task=tokens)")
    add("--token-dtype", default="uint16", choices=["uint8", "uint16", "int32", "int64"])
    add("--validation-tokens", default="", help="separate validation token file for periodic evaluation")
    add("--split", choices=["train", "val", "test"], default="val", help="enwik8 evaluation split")
    add("--vocab-size", type=int, default=128)
    add("--dim", type=int, default=256)
    add("--heads", type=int, default=4)
    add("--block-size", type=int, default=16)
    add("--arch", default="Lx2,G")
    add("--ffn-ratio", type=float, default=4.)
    add("--topk", type=int, default=None, help="inference top-k; training is always dense")
    add("--length", type=int, default=512)
    add("--batch-size", type=int, default=1, help="microbatch per data-parallel replica")
    add("--grad-accum", type=int, default=1)
    add("--context-parallel", default="auto", help="auto or a divisor of heads and WORLD_SIZE")
    add("--precision", choices=["fp32", "bf16"], default="bf16")
    add("--backend", choices=["auto", "torch", "triton"], default="auto")
    add("--checkpoint-chunks", action=argparse.BooleanOptionalAction, default=True)
    add("--activation-offload", action="store_true", help="offload autograd saved tensors to pinned host RAM")
    add("--optimizer-shard", action=argparse.BooleanOptionalAction, default=True)
    add("--encoder-chunk", type=int, default=1024)
    add("--query-chunk", type=int, default=128, help="local training queries per chunk")
    add("--loss-chunk", type=int, default=128)
    add("--steps", type=int, default=1000)
    add("--eval-every", type=int, default=0, help="periodic evaluation interval; 0 disables it")
    add("--stop-exact", type=float, default=None, help="save and stop when periodic exact accuracy reaches threshold")
    add("--lr", type=float, default=3.e-4)
    add("--weight-decay", type=float, default=.01)
    add("--clip-grad", type=float, default=1.)
    add("--seed", type=int, default=0)
    add("--log-every", type=int, default=10)
    add("--save-every", type=int, default=0)
    add("--save", default="runs/stack.pt")
    add("--resume", default="")
    add("--weights-only", action="store_true", help="load weights, reset optimizer and RNG")
    add("--log", default="", help="rank-zero JSONL path")
    add("--device", choices=["cuda", "cpu"], default="cuda")
    add("--memory-fraction", type=float, default=.8)
    add("--allow-over-budget", action="store_true", help="override a conservative training estimate; never changes semantics")
    add("--npairs", type=int, default=16)
    add("--npairs-density", type=float, default=0., help="MQAR pairs per token, overriding fixed npairs")
    add("--nqueries", type=int, default=4)
    add("--nkeytoks", type=int, choices=[1, 2], default=1)
    add("--eval-batches", type=int, default=1)
    add("--eval-positions", type=int, default=0, help="0=task mask; otherwise uniformly sampled query positions")
    add("--cache", default="auto", choices=["auto", "cuda", "cpu", "disk"])
    add("--cache-dir", default="")
    add("--page-tokens", type=int, default=65536)
    add("--inference-query-chunk", type=int, default=16)
    add("--workspace-mb", type=int, default=256)
    add("--max-query-states-mb", type=int, default=256)
    add("--output", default="", help="eval JSON output, written by rank zero")


def parser():
    p = argparse.ArgumentParser(
        prog="python -m st", description="Stack experiments: one command entry point.",
        epilog="Start: python -m st benchmark --help | Guide: README.md and docs/COMMANDS.md",
        allow_abbrev=False,
    )
    sub = p.add_subparsers(dest="command", required=True, title="commands")
    p.experiments = {}
    for name in ("plan", "train", "eval"):
        child = sub.add_parser(name, help=COMMANDS[name], description=COMMANDS[name], allow_abbrev=False)
        _experiment_options(child, name)
        p.experiments[name] = child
    # These commands own distinct option sets; reuse their parsers in help.
    from .tools.benchmark import parser as benchmark_parser
    from .tools.validate import parser as validate_parser
    sub.add_parser("benchmark", help=COMMANDS["benchmark"], parents=[benchmark_parser()], add_help=False, allow_abbrev=False)
    sub.add_parser("validate", help=COMMANDS["validate"], parents=[validate_parser()], add_help=False, allow_abbrev=False)
    return p


def parse_args(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.command not in p.experiments:
        return p.parse_args(argv)
    experiment = p.experiments[preliminary.command]
    if preliminary.config:
        with open(preliminary.config) as f:
            config = json.load(f)
        if not isinstance(config, dict) or any(not isinstance(config.get(k, {}), dict) for k in ("model", "runtime")):
            p.error("config must contain model/runtime JSON objects")
        if set(config)-{"model", "runtime"}:
            raise ValueError("config root accepts only model and runtime")
        defaults = dict(config.get("model", {}), **config.get("runtime", {}))
        known = ({a.dest for a in experiment._actions} | set(experiment._defaults)) - {"command", "config", "help"}
        if set(defaults)-known:
            raise ValueError(f"unknown config keys: {sorted(set(defaults)-known)}")
        experiment.set_defaults(**defaults)
    args = p.parse_args(argv)
    # argparse does not validate choices or non-string defaults supplied by JSON.
    for action in experiment._actions:
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
    if not argv:
        parser().print_help()
        return
    if argv and argv[0] == "validate":
        from .tools.validate import main as validate
        return validate(argv[1:])
    if argv and argv[0] == "benchmark":
        from .tools.benchmark import main as benchmark
        return benchmark(argv[1:])
    args = parse_args(argv)
    from .runtime.engine import run
    return run(args)


def entrypoint():
    """Console scripts must not use a result dictionary as an exit status."""
    main()
