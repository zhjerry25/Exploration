#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Synthetic tasks
#
# Passkey Retrival + MQAR
# 
# Model: stack, arch=Lx2,G, d=256, 2.4M Params
#
# MQAR ignition from 128 → 512 training；passkey 512 training
# 
# Zero-shot extrapolation 4096 → 65536 → 1M → 2M → 4M → 8M → 16M
#
# MQAR：npairs=16 + nqueries=16 (single token key)
#
# Key-value density dilutes as length increase
#
# Values are numbers from 0-9
#
# fp32 for evaluation is used, up to 96GB memory is required
#
# 3 seeds are used
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p runs
say() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

for S in 0 1 2; do

  # ── S1 MQAR ──

  say "S1 MQAR seed=$S"

  say "S1 MQAR 128 ignition"

  python -m st.train --task mqar --n 128 --npairs 16 --nqueries 16 \
  --steps 3000 --bs 64 --lr 1e-3 --bf16 --seed $S --save runs/s1_mqar128_s$S.pt

  say "S1 MQAR 512 training"

  python -m st.train --task mqar --n 512 --npairs 16 --nqueries 16 \
  --steps 6000 --bs 64 --lr 5e-4 --bf16 --seed $S \
  --resume_weights_only runs/s1_mqar128_s$S.pt --save runs/s1_mqar512_s$S.pt

  say "S1 MQAR zero-shot extrapolation"

  for N in 512 4096 65536 1000000 2000000 4000000 8000000 16000000; do
    python -m st.train --task mqar --n $N --npairs 16 --nqueries 16 \
      --eval_only --resume runs/s2_mqar512_s$S.pt --bs 1 --seed $S
  done

  # ── S2 passkey ──

  say "S2 passkey 512 training"

  say "A passkey seed=$S"

  python -m st.train --task passkey --n 512 --steps 3000 --bs 64 --lr 5e-4 \
    --bf16 --seed $S --save runs/s2_passkey512_s$S.pt
  
  say "S2 passkey zero-shot extrapolation"

  for N in 512 4096 65536 1000000 2000000 4000000 8000000 16000000; do
    python -m st.train --task passkey --n $N \
      --eval_only --resume runs/s2_passkey512_s$S.pt --bs 1 --seed $S
  done
done

say "S group experiments over"

# optional shutdown
/usr/bin/shutdown