#!/usr/bin/env bash
set -euo pipefail
GPUS=${GPUS:-1}
REPORT_DIR=${REPORT_DIR:-runs/smoke}
mkdir -p "$REPORT_DIR"

# First run remote_validate.sh. This smoke stage exercises the real driver,
# optimizer, gradient accumulation, checkpoints and reload on small inputs.
torchrun --standalone --nproc_per_node="$GPUS" -m st.run train \
  --config configs/stack_2m.json --task random --length 512 --steps 4 \
  --grad-accum 2 --save-every 2 --save "$REPORT_DIR/model.pt" \
  --log-every 1 --log "$REPORT_DIR/train.jsonl" 2>&1 | tee "$REPORT_DIR/train.log"

python -m st.run eval --resume "$REPORT_DIR/model.pt" --task random \
  --length 2048 --eval-positions 16 --cache cpu --output "$REPORT_DIR/eval.json" \
  2>&1 | tee "$REPORT_DIR/eval.log"

python -m st.benchmark --operation dense --backend eager --length 512 \
  --output "$REPORT_DIR/eager.json"
python -m st.benchmark --operation dense --backend triton --length 512 \
  --output "$REPORT_DIR/triton.json"
python -m st.benchmark --operation sparse --backend triton --length 65536 \
  --output "$REPORT_DIR/sparse.json"
