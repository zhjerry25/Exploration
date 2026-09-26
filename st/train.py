"""Train/eval driver for the stack model.

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
  # enwik8 LM (data/enwik8 bundled)
  python -m st.train --task lm --n 4096 --steps 15000 --bs 16 --lr 5e-4 \
      --save runs/lm.pt
"""
import argparse
import contextlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from . import data
from . import lmdata
from .stack_model import StackModel

VOCABS = {"passkey": data.VOCAB, "copying": data.VOCAB, "mqar": data.VOCAB,
          "lm": 256}


def get_device():
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
    m = StackModel(vocab, dim=args.d, heads=args.heads, block_size=args.b,
                   topk=getattr(args, "read_m", 64),
                   local_layers=getattr(args, "local_layers", 2))
    return m.to(device)


def _emb_norm(model):
    return model.embedding.weight.norm().item()


BATCHERS = {"passkey": data.passkey_batch, "copying": data.copying_batch,
            "mqar": data.mqar_batch, "lm": lmdata.lm_batch}


def make_batch(args, g, device, n=None, split="train"):
    if args.task == "lm":
        return lmdata.lm_batch(args.bs, n or args.n, g, device, split=split)
    if args.task == "mqar":
        return data.mqar_batch(args.bs, n or args.n, g, device,
                               n_pairs=getattr(args, "npairs", 16),
                               n_queries=getattr(args, "nqueries", 4))
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


def posloss_mod_b(model, args, g_eval, device, bb):
    """Mean bpc by position index mod bb (one fresh val batch)."""
    with torch.no_grad(), amp_ctx(args, device):
        idx, tgt, mask, _ = make_batch(args, g_eval, device, split="val")
        lgs = model(idx)
        ce = F.cross_entropy(lgs.reshape(-1, lgs.shape[-1]).float(),
                             tgt.reshape(-1), reduction="none")
        ce = ce.view(tgt.shape[0], tgt.shape[1]).mean(0)
        return [round(x, 3) for x in ce.view(-1, bb).mean(0).div(0.6931).tolist()]


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
    with torch.no_grad(), amp_ctx(args, device):
        for _ in range(batches):
            idx, tgt, mask, pos = make_batch(args, g_eval, device, split="val")
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
        # (starved for cross-block info) or uniform (capacity-bound)?
        rec["posloss_mod_b"] = posloss_mod_b(model, args, g_eval, device,
                                             args.b)
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
    model = build(args, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model=stack n={args.n} params={n_params/1e6:.2f}M "
          f"device={device}", flush=True)
    ema = None
    if args.ema > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    start_step = 0
    if args.resume_weights_only:
        # for switching to a longer sequence length: load weights, fresh
        # optimizer/EMA, restart step counter and schedule
        ck = torch.load(args.resume_weights_only, map_location=device,
                        weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"weights-only resume from {args.resume_weights_only}: "
              f"{len(missing)} new / {len(unexpected)} skipped params", flush=True)
        for k in missing:
            print(f"  + new: {k}", flush=True)
    elif args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
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
    t0, last_t = time.time(), time.time()
    for step in range(start_step, args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, args.steps, args.lr)
        idx, tgt, mask, pos = make_batch(args, g_train, device)
        model.train()
        with amp_ctx(args, device):
            loss, pd, em, _, _ = loss_and_acc(model, idx, tgt, mask)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
        if step % 100 == 0:
            now = time.time()
            tok_s = int(args.bs * args.n * 100 / max(now - last_t, 1e-9)) if step > 0 else 0
            last_t = now
            rec = dict(step=step, loss=round(loss.item(), 4), digit_acc=round(pd, 4),
                       exact=round(em, 4), tok_s=tok_s, sec=round(now - t0, 1),
                       gnorm=round(float(gnorm), 3),
                       emb_n=round(_emb_norm(model), 3))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if args.save and args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            save_ckpt(args.save, model, opt, ema, args, step)
        if step % args.eval_every == 0 or step == args.steps - 1:
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            model.eval()
            rec = dict(step=step, **evaluate(model, args, g_eval, device, 4))
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
    ap.add_argument("--task", default="passkey",
                    choices=["passkey", "copying", "mqar", "lm"])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--b", type=int, default=16)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA eval decay, 0=off")
    ap.add_argument("--bf16", action="store_true", help="CUDA bf16 autocast")
    ap.add_argument("--save", default="", help="checkpoint path (.pt)")
    ap.add_argument("--ckpt_every", type=int, default=0, help="periodic save interval")
    ap.add_argument("--resume", default="", help="resume from checkpoint path")
    ap.add_argument("--resume_weights_only", default="",
                    help="lenient weight resume (fresh opt/schedule, step 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--read_m", type=int, default=64,
                    help="top-m blocks popped per head")
    ap.add_argument("--local_layers", type=int, default=2,
                    help="depth of the local halo encoder")
    ap.add_argument("--nqueries", type=int, default=4, help="mqar queries per sequence")
    ap.add_argument("--npairs", type=int, default=16, help="mqar pairs per sequence")
    ap.add_argument("--stop_exact", type=float, default=None,
                    help="early stop + save when eval_exact >= this (0..1)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval_only", action="store_true",
                    help="load --resume checkpoint, eval once at args.n, exit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = f"stack_{args.task}_n{args.n}_s{args.seed}"
    device = get_device()
    if args.selftest:
        leak_test()
        overfit_test(device)
        return
    train(args, device)


if __name__ == "__main__":
    main()
