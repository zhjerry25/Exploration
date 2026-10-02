#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# ─────────────────────────────────────────────────────────────────────────────
# Dual-side scaling matrix on enwik8 (write-side L × read-side G)
#
# Stack: d=256, arch=Lx{L},Gx{G}, independent weights only.
# Per-cell baseline: matched params, layers = round((12L+10G+2)/12).
#
# Protocol: length=4096, batch-size 16, 8000 steps, lr 5e-4, bf16, seed 0.
# Eval: 4096 (train length, periodic log) + zero-shot 16384/65536, fp32.
# 12 cells, 2 workers (one per GPU), ~5-7h on 2x96GB.
# Report: python scripts/experiments/scaling_matrix_report.py
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p runs
say() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

CELLS=("L,G" "Lx2,G" "Lx4,G" "Lx8,G" "L,Gx2" "Lx2,Gx2" "Lx4,Gx2" "Lx8,Gx2" "L,Gx4" "Lx2,Gx4" "Lx4,Gx4" "Lx8,Gx4")
LAYERS=(2 3 5 9 3 4 6 10 5 6 8 13)
TAGS=(l1g1 l2g1 l4g1 l8g1 l1g2 l2g2 l4g2 l8g2 l1g4 l2g4 l4g4 l8g4)

run_cell() {
  local I=$1 GPU=$2
  local A=${CELLS[$I]} BL=${LAYERS[$I]} T=${TAGS[$I]}
  say "cell $T arch=$A baseline_layers=$BL gpu=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU python -m st train --task enwik8 --length 4096 --arch "$A" \
    --batch-size 16 --steps 8000 --lr 5e-4 --precision bf16 \
    --no-checkpoint-chunks --no-optimizer-shard \
    --save runs/sc_$T.pt --eval-every 500 --eval-batches 4 --log runs/sc_$T.jsonl
  for N in 16384 65536; do
    CUDA_VISIBLE_DEVICES=$GPU python -m st eval --task enwik8 --length $N \
      --resume runs/sc_$T.pt --batch-size 2 --precision fp32 --eval-batches 4 --log runs/sc_$T.jsonl
  done
  CUDA_VISIBLE_DEVICES=$GPU python -m st train --model baseline --layers $BL --task enwik8 --length 4096 \
    --batch-size 16 --steps 8000 --lr 5e-4 --precision bf16 \
    --no-checkpoint-chunks --no-optimizer-shard \
    --save runs/sc_${T}_base.pt --eval-every 500 --eval-batches 4 --log runs/sc_${T}_base.jsonl
  for N in 16384 65536; do
    CUDA_VISIBLE_DEVICES=$GPU python -m st eval --model baseline --task enwik8 --length $N \
      --resume runs/sc_${T}_base.pt --batch-size 2 --precision fp32 --eval-batches 4 --log runs/sc_${T}_base.jsonl
  done
}

worker() { local GPU=$1; shift; for I in "$@"; do run_cell $I $GPU; done; }

worker 0 0 1 2 3 4 5 &
worker 1 6 7 8 9 10 11 &
wait
say "scaling matrix done"
