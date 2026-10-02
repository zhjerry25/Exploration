#!/usr/bin/env bash
set -euo pipefail

# Short, reproducible throughput comparison for the fair Stack/Baseline pair.
# Run on the CUDA host from the repository root.  It intentionally uses a
# small number of optimizer steps; increase STEPS only after validation.
GPUS=${GPUS:-2}
LENGTH=${LENGTH:-8192}
STEPS=${STEPS:-24}
LOG_EVERY=${LOG_EVERY:-8}
CHECKPOINT=${CHECKPOINT:-0}
REPORT_DIR=${REPORT_DIR:-runs/benchmarks/camera-ready}
mkdir -p "$REPORT_DIR"

checkpoint_args=(--no-checkpoint-chunks)
if [[ "$CHECKPOINT" == 1 ]]; then
  checkpoint_args=(--checkpoint-chunks)
fi
common=(--task random --length "$LENGTH" --batch-size 1 --steps "$STEPS"
  --grad-accum 1 --log-every "$LOG_EVERY" --save '' --precision bf16
  "${checkpoint_args[@]}" --backend auto)

torchrun --standalone --nproc_per_node="$GPUS" -m st train \
  --model stack --config configs/stack_2m.json "${common[@]}" \
  > "$REPORT_DIR/stack.jsonl" 2>&1

torchrun --standalone --nproc_per_node="$GPUS" -m st train \
  --model baseline --vocab-size 128 --dim 256 --heads 4 --layers 3 \
  --flash-attention auto "${common[@]}" \
  > "$REPORT_DIR/baseline.jsonl" 2>&1

python -m st benchmark --compare --operation dense --length 512 \
  --output "$REPORT_DIR/dense-operator.json"

echo "Reports written to $REPORT_DIR"
