#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# ─────────────────────────────────────────────────────────────────────────────
# Synthetic tasks
#
# Passkey Retrival + MQAR
# 
# Model: stack, arch=Lx2,G, d=256, 2.4M Params
#
# MQAR ignition from 128 → 512 training；passkey 512 training
# 
# Zero-shot extrapolation 4096 → 65536 → 1M → 2M → 4M → 8M → 12M
#
# MQAR：npairs=16 + nqueries=16 (single token key)
#
# Key-value density dilutes as length increase
#
# Values are numbers from 0-9
#
# fp32 evaluation; the unified planner selects GPU/CPU KV storage as needed.
#
# 3 seeds are used
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p runs
say() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

for S in 0 1 2; do

  # ── S1 MQAR ──

  say "S1 MQAR seed=$S"

  say "S1 MQAR 128 ignition"

  python -m st train --task mqar --length 128 --npairs 16 --nqueries 16 \
  --steps 3000 --batch-size 64 --lr 1e-3 --precision bf16 --seed $S --save runs/s1_mqar128_s$S.pt \
  --eval-every 500 --eval-batches 4 --log runs/s1_mqar128_s$S.jsonl

  say "S1 MQAR 512 training"

  python -m st train --task mqar --length 512 --npairs 16 --nqueries 16 \
  --steps 6000 --batch-size 64 --lr 5e-4 --precision bf16 --seed $S \
  --weights-only --resume runs/s1_mqar128_s$S.pt --save runs/s1_mqar512_s$S.pt \
  --eval-every 500 --eval-batches 4 --log runs/s1_mqar512_s$S.jsonl

  say "S1 MQAR zero-shot extrapolation"

  for N in 512 4096 65536 1000000 2000000 4000000 8000000 12000000; do
    python -m st eval --task mqar --length $N --npairs 16 --nqueries 16 \
      --resume runs/s1_mqar512_s$S.pt --batch-size 1 --seed $S --precision fp32 \
      --eval-batches 8 --log runs/s1_mqar512_s$S.eval.jsonl
  done

  # ── S2 passkey ──

  say "S2 passkey 512 training"

  say "A passkey seed=$S"

  python -m st train --task passkey --length 512 --steps 3000 --batch-size 64 --lr 5e-4 \
    --precision bf16 --seed $S --save runs/s2_passkey512_s$S.pt \
    --eval-every 500 --eval-batches 4 --log runs/s2_passkey512_s$S.jsonl
  
  say "S2 passkey zero-shot extrapolation"

  for N in 512 4096 65536 1000000 2000000 4000000 8000000 12000000; do
    python -m st eval --task passkey --length $N \
      --resume runs/s2_passkey512_s$S.pt --batch-size 1 --seed $S --precision fp32 \
      --eval-batches 8 --log runs/s2_passkey512_s$S.eval.jsonl
  done
done

say "S group experiments over"

# No automatic shutdown.
