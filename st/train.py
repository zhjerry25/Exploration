"""Train/eval driver for the stack model and the dense baseline.

Doctrine: TRAIN DENSE, INFER SPARSE. Training runs must be exactly dense
(read_m >= n/b, enforced at startup) — the sparse top-m read path exists
for inference only, where it activates zero-shot at long n.

Usage:
  python -m st.train --selftest          # leak test (must be 0 diff) + overfit
  # synthetic: ignite small, transfer big
  python -m st.train --task mqar --n 128 --npairs 16 --nqueries 16 \
      --steps 3000 --bs 64 --lr 1e-3 --stop_exact 0.99 --save runs/st128.pt
  python -m st.train --task mqar --n 512 --npairs 16 --nqueries 16 \
      --steps 2000 --bs 64 --lr 5e-4 --stop_exact 0.99 \
      --resume_weights_only runs/st128.pt --save runs/st512.pt
  python -m st.train --task mqar --n 4096 --eval_only --resume runs/st512.pt
  python -m st.train --task mqar --n 16384 --eval_only --resume runs/st512.pt --bs 8
  python -m st.train --task passkey --n 512 --steps 1500 --bs 64 --lr 5e-4 \
      --stop_exact 0.99 --save runs/stpk512.pt
  python -m st.train --task passkey --n 65536 --eval_only --resume runs/stpk512.pt --bs 4
  # enwik8 LM, dense @512 (read_m=64 >= 512/16); sparse only in zero-shot eval
  python -m st.train --task lm --n 512 --steps 30000 --bs 32 --lr 5e-4 \
      --bf16 --save runs/lm512.pt
  python -m st.train --task lm --n 65536 --eval_only --resume runs/lm512.pt --bs 2
  # dual-side scaling (arch spec: LxN independent / (X)xN weight-shared)
  python -m st.train --task lm --n 512 --arch "Lx2,(G)x4" --bf16 ...
  # dense transformer baseline, same params and layer passes as "Lx2,G"
  python -m st.train --model baseline --layers 3 --task lm --n 512 ...
  # density-scaled hard MQAR: pairs grow with n, two-token keys
  python -m st.train --task mqar --n 512 --npairs_density 0.03125 --nkeytoks 2 ...
  # eval resume inherits the checkpoint's architecture (arch/b/d/heads);
  # --n (eval length) and --read_m (top-m) are the eval-time knobs
  python -m st.train --task lm --n 65536 --eval_only --resume runs/lm512.pt --bs 2
"""
import argparse
import contextlib
import json
import math
import os
import re
import time

import torch
import torch.nn.functional as F

from . import data
from . import lmdata
from .baseline import BaselineModel
from .stack_model import StackModel

VOCABS = {"passkey": data.VOCAB, "copying": data.VOCAB, "mqar": data.VOCAB,
          "lm": 256}


def get_device(requested=""):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_ctx(args, device):
    if getattr(args, "bf16", False) and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build(args, device="cpu"):
    vocab = VOCABS.get(getattr(args, "task", "passkey"), data.VOCAB)
    if getattr(args, "model", "stack") == "baseline":
        m = BaselineModel(vocab, dim=args.d, heads=args.heads,
                          layers=getattr(args, "layers", 3),
                          ffn_ratio=getattr(args, "ffn_ratio", 4))
    else:
        m = StackModel(vocab, dim=args.d, heads=args.heads, block_size=args.b,
                       topk=getattr(args, "read_m", None) or 64,
                       arch=getattr(args, "arch", None) or "Lx2,G",
                       ffn_ratio=getattr(args, "ffn_ratio", 4))
    return m.to(device)


def _emb_norm(model):
    return model.embedding.weight.norm().item()


BATCHERS = {"passkey": data.passkey_batch, "copying": data.copying_batch,
            "mqar": data.mqar_batch, "lm": lmdata.lm_batch}


def make_batch(args, g, device, n=None, split="train"):
    if args.task == "lm":
        return lmdata.lm_batch(args.bs, n or args.n, g, device, split=split)
    if args.task == "mqar":
        nn = n or args.n
        npairs = data.resolve_npairs(nn, getattr(args, "npairs", 16),
                                     getattr(args, "npairs_density", 0.0))
        return data.mqar_batch(args.bs, nn, g, device, n_pairs=npairs,
                               n_queries=getattr(args, "nqueries", 4),
                               key_tokens=getattr(args, "nkeytoks", 1))
    return BATCHERS[args.task](args.bs, n or args.n, g, device)


def loss_and_acc(model, idx, tgt, mask):
    """CE on masked positions; the model receives the loss mask so the
    push/pop machinery runs only at supervised positions. Returns (loss,
    digit_acc, exact, hit_vec, col_acc) with col_acc the per-masked-column
    accuracy (diagnostic for passkey digits)."""
    logits = model(idx, sup=mask)
    loss = F.cross_entropy(logits[mask], tgt[mask])
    lp, tp = logits[mask], tgt[mask]
    hit = (lp.argmax(-1) == tp).view(idx.shape[0], -1)
    hit_vec = hit.all(1)
    return (loss, hit.float().mean().item(), hit_vec.float().mean().item(),
            hit_vec, hit.float().mean(0))


def posloss_diag(model, args, g_eval, device, bb, bins=16):
    """Per-position loss diagnostics on one fresh val batch: mean bpc by
    position index mod bb (block-boundary pattern) plus the log-binned
    bpc-by-absolute-position curve (the state-dilution probe: a healthy
    long-context model's loss should FALL with position)."""
    with torch.no_grad(), amp_ctx(args, device):
        idx, tgt, mask, _ = make_batch(args, g_eval, device, split="val")
        lgs = model(idx)
        ce = F.cross_entropy(lgs.reshape(-1, lgs.shape[-1]).float(),
                             tgt.reshape(-1), reduction="none")
        ce = ce.view(tgt.shape[0], tgt.shape[1]).mean(0).div(0.6931)  # [n] bpc
        n = ce.shape[0]
        mod_b = None
        if n % bb == 0:
            mod_b = [round(x, 3) for x in ce.view(-1, bb).mean(0).tolist()]
        edges = torch.logspace(0, math.log10(n), bins + 1).round().long()
        edges = edges.unique().clamp(min=1).tolist()
        curve, lo = [], 0
        for hi in edges:
            if hi > lo:
                curve.append(round(ce[lo:hi].mean().item(), 3))
            lo = hi
        return mod_b, curve


def save_ckpt(path, model, opt, ema, args, step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "ema": ema, "args": vars(args), "step": step}, path)


def leak_test():
    """Shuffle the last 10% of input tokens; logits over the first 90% must
    not change. Run on CPU for determinism. The single most important test in
    this codebase."""
    n, cutoff = 1024, 1024 - 102
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, data.VOCAB, (2, n), generator=g)
    idx2 = idx.clone()
    idx2[:, cutoff:] = torch.randint(0, data.VOCAB, (2, n - cutoff), generator=g)
    args = argparse.Namespace(d=64, heads=2, b=16)
    model = build(args, device="cpu").eval()
    with torch.no_grad():
        l1, l2 = model(idx), model(idx2)
    err = (l1[:, :cutoff] - l2[:, :cutoff]).abs().max().item()
    ok = err < 1e-4
    print(f"[leak] stack max-logit-diff on past positions: {err:.2e} "
          f"-> {'OK' if ok else 'CAUSAL LEAK!'}")
    assert ok, "causal leak detected; do not train until fixed"


def overfit_test(device):
    """A single fixed passkey batch (n=512) must be memorizable within a few
    hundred steps."""
    args = argparse.Namespace(task="passkey", n=512, d=128, heads=4, b=16,
                              bs=32)
    torch.manual_seed(0)
    model = build(args, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    idx, tgt, mask, _ = make_batch(args, g, device)
    model.train()
    for step in range(400):
        loss, pd, em, _, _ = loss_and_acc(model, idx, tgt, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 399:
            print(f"[overfit] step {step:4d} loss {loss.item():.4f} "
                  f"digit-acc {pd:.3f} exact {em:.3f}")
    print("[overfit] done (expect exact ~1.0)")


def lr_at(step, total, base):
    warm = max(1, total // 20)
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def evaluate(model, args, g_eval, device, batches):
    """Shared eval body: loss/exact/depth/bpc over fresh val batches."""
    ems, pds, hits, poss, lss, cas = [], [], [], [], [], []
    first_batch = None
    with torch.no_grad(), amp_ctx(args, device):
        for i in range(batches):
            idx, tgt, mask, pos = make_batch(args, g_eval, device, split="val")
            if i == 0:
                first_batch = (idx, mask, pos)
            l, pd, em, hv, ca = loss_and_acc(model, idx, tgt, mask)
            lss.append(l.item())
            ems.append(em)
            pds.append(pd)
            cas.append(ca)
            if pos is not None:
                hits.append(hv.cpu())
                poss.append(pos.cpu())
    rec = dict(eval_exact=round(sum(ems) / len(ems), 4),
               eval_digit=round(sum(pds) / len(pds), 4))
    if args.task == "passkey":
        rec["pos_acc"] = [round(x, 3) for x in
                          torch.stack(cas).mean(0).tolist()]
    if args.task == "lm":
        rec["bpc"] = round(sum(lss) / len(lss) / 0.6931, 4)
        # per-position loss by index mod b: is the gap at block boundaries
        # (starved for cross-block info) or uniform (capacity-bound)? plus
        # the log-binned bpc-by-position curve (state-dilution probe).
        mod_b, curve = posloss_diag(model, args, g_eval, device, args.b)
        rec["posloss_mod_b"] = mod_b
        rec["posloss_bpc_curve"] = curve
    if isinstance(model, StackModel):
        idx0, mask0, pos0 = first_batch
        rec.update(model.diagnose(idx0, mask0,
                                  pos0 if args.task == "passkey" else None))
    if poss:
        h, p = torch.cat(hits).float(), torch.cat(poss)
        rec["depth_exact"] = [
            round(h[(p >= qi * args.n // 4) & (p < (qi + 1) * args.n // 4)]
                  .mean().item(), 3)
            if ((p >= qi * args.n // 4) & (p < (qi + 1) * args.n // 4)).any()
            else -1
            for qi in range(4)
        ]
    return rec


def train(args, device):
    torch.manual_seed(args.seed)
    ck = None
    if args.resume_weights_only or args.resume:
        # load early: the checkpoint is authoritative for model structure
        # (arch/b/d/heads/layers/ffn_ratio), CLI structural flags only apply
        # to fresh training. n stays a CLI knob (eval length), and read_m
        # stays an inference knob (default: the checkpoint's value).
        ck = torch.load(args.resume_weights_only or args.resume,
                        map_location="cpu", weights_only=False)
        trained = ck.get("args", {})
        for k in ("model", "arch", "layers", "d", "heads", "b", "ffn_ratio"):
            if trained.get(k) is not None:
                setattr(args, k, trained[k])
        if args.read_m is None:
            args.read_m = trained.get("read_m")
    if args.read_m is None:
        args.read_m = 64
    model = build(args, device=device)
    if isinstance(model, StackModel) and not args.eval_only:
        # dense-train doctrine gate: training must be exactly dense, so the
        # sparse top-m machinery only ever runs at inference
        groups = -(-args.n // args.b)
        if args.read_m < groups:
            raise SystemExit(
                f"dense-train doctrine violated: read_m={args.read_m} < "
                f"{groups} blocks at n={args.n}, b={args.b}. Raise --read_m "
                f"or train shorter; sparsity belongs to inference only.")
    n_params = sum(p.numel() for p in model.parameters())
    if isinstance(model, StackModel):
        passes = len(model.local) + len(model.reads)
        desc = f"model=stack arch={model.arch} passes={passes}"
    else:
        desc = f"model=baseline layers={len(model.blocks)}"
    print(f"{desc} n={args.n} params={n_params/1e6:.2f}M "
          f"device={device}", flush=True)
    ema = None
    if args.ema > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    start_step = 0
    if args.resume_weights_only:
        # for switching to a longer sequence length: load weights, fresh
        # optimizer/EMA, restart step counter and schedule
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"weights-only resume from {args.resume_weights_only}: "
              f"{len(missing)} new / {len(unexpected)} skipped params",
              flush=True)
        for k in missing:
            print(f"  + new: {k}", flush=True)
    elif args.resume:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema is not None and ck.get("ema") is not None:
            ema = ck["ema"]
        start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}", flush=True)
    g_eval = torch.Generator().manual_seed(args.seed + 200)
    if args.eval_only:
        assert args.resume, "--eval_only requires --resume"
        if ck.get("ema") is not None:
            model.load_state_dict(ck["ema"])
        model.eval()
        rec = dict(n=args.n, **evaluate(model, args, g_eval, device, 8))
        print(json.dumps(rec), flush=True)
        return
    g_train = torch.Generator().manual_seed(args.seed + 100)
    os.makedirs("runs", exist_ok=True)
    log = open(f"runs/{args.tag}.jsonl", "a")
    if getattr(args, "prof", 0):
        model._prof = True
    t0, last_t = time.time(), time.time()
    eval_acc = 0.0  # eval seconds inside the current tok_s window (excluded)
    for step in range(start_step, args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, args.steps, args.lr)
        idx, tgt, mask, pos = make_batch(args, g_train, device)
        model.train()
        with amp_ctx(args, device):
            loss, pd, em, _, _ = loss_and_acc(model, idx, tgt, mask)
        opt.zero_grad()
        tb = time.time()
        loss.backward()
        bwd_dt = time.time() - tb
        if getattr(args, "prof", 0) and hasattr(model, "_prof_stats"):
            model._prof_stats["bwd_s"] = round(bwd_dt, 3)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
        if step % 100 == 0:
            now = time.time()
            train_dt = now - last_t - eval_acc
            tok_s = int(args.bs * args.n * 100 / max(train_dt, 1e-9)) if step > 0 else 0
            last_t, eval_acc = now, 0.0
            rec = dict(step=step, loss=round(loss.item(), 4), digit_acc=round(pd, 4),
                       exact=round(em, 4), tok_s=tok_s, sec=round(now - t0, 1),
                       gnorm=round(float(gnorm), 3),
                       emb_n=round(_emb_norm(model), 3))
            if getattr(args, "prof", 0) and hasattr(model, "_prof_stats"):
                rec.update(model._prof_stats)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if args.save and args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            save_ckpt(args.save, model, opt, ema, args, step)
        if step % args.eval_every == 0 or step == args.steps - 1:
            te = time.time()
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            model.eval()
            rec = dict(step=step, **evaluate(model, args, g_eval, device, 4))
            eval_dt = time.time() - te
            eval_acc += eval_dt
            rec["eval_s"] = round(eval_dt, 1)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if (args.stop_exact is not None
                    and rec.get("eval_exact", 0) >= args.stop_exact):
                if args.save:
                    save_ckpt(args.save, model, opt, ema, args, step)
                print(f"[early-stop] eval_exact {rec['eval_exact']} >= "
                      f"{args.stop_exact} at step {step}, saved", flush=True)
                log.close()
                return
            if backup is not None:
                model.load_state_dict(backup)
    if args.save:
        save_ckpt(args.save, model, opt, ema, args, args.steps - 1)
        print(f"saved {args.save}", flush=True)
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stack", choices=["stack", "baseline"])
    ap.add_argument("--task", default="passkey",
                    choices=["passkey", "copying", "mqar", "lm"])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--b", type=int, default=16)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--arch", default=None,
                    help="stack layer spec, e.g. 'Lx2,G' (default), "
                         "'Lx2,(G)x4' (read cycling), 'Lx4,Gx2'")
    ap.add_argument("--layers", type=int, default=3, help="baseline depth")
    ap.add_argument("--ffn_ratio", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA eval decay, 0=off")
    ap.add_argument("--bf16", action="store_true", help="CUDA bf16 autocast")
    ap.add_argument("--prof", type=int, default=0,
                    help="log encode/read/head phase seconds every 100 steps")
    ap.add_argument("--device", default="",
                    help="force device (cuda/cpu/mps); default auto")
    ap.add_argument("--save", default="", help="checkpoint path (.pt)")
    ap.add_argument("--ckpt_every", type=int, default=0, help="periodic save interval")
    ap.add_argument("--resume", default="", help="resume from checkpoint path")
    ap.add_argument("--resume_weights_only", default="",
                    help="lenient weight resume (fresh opt/schedule, step 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--read_m", type=int, default=None,
                    help="top-m blocks popped per head (inference; default 64 "
                         "or the checkpoint's own value. Training requires "
                         "read_m >= n/b: dense-train doctrine)")
    ap.add_argument("--nqueries", type=int, default=4, help="mqar queries per sequence")
    ap.add_argument("--npairs", type=int, default=16, help="mqar pairs per sequence")
    ap.add_argument("--npairs_density", type=float, default=0.0,
                    help="mqar pairs per token (overrides --npairs, scales with n)")
    ap.add_argument("--nkeytoks", type=int, default=1, choices=[1, 2],
                    help="mqar key length in tokens (2 -> 4096 distinct keys)")
    ap.add_argument("--stop_exact", type=float, default=None,
                    help="early stop + save when eval_exact >= this (0..1)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval_only", action="store_true",
                    help="load --resume checkpoint, eval once at args.n, exit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.tag is None:
        # unique per configuration: concurrent runs never share a log file
        parts = [args.model, args.task, f"n{args.n}"]
        if args.model == "stack":
            arch = args.arch or "Lx2,G"
            parts += [re.sub(r"[^A-Za-z0-9]+", "", arch), f"b{args.b}"]
        else:
            parts += [f"L{args.layers}"]
        parts.append(f"s{args.seed}")
        args.tag = "_".join(parts)
    device = get_device(args.device)
    if args.selftest:
        leak_test()
        overfit_test(device)
        return
    train(args, device)


if __name__ == "__main__":
    main()
