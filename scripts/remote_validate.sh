#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the remote NVIDIA machine. No installs,
# downloads, shutdowns, or destructive operations are performed.
GPUS=${GPUS:-1}
REPORT_DIR=${REPORT_DIR:-runs/validation}
mkdir -p "$REPORT_DIR"
python -m st validate --suite reference --output "$REPORT_DIR/reference.json" 2>&1 | tee "$REPORT_DIR/reference.log"
python -m st validate --suite cuda --output "$REPORT_DIR/cuda.json" 2>&1 | tee "$REPORT_DIR/cuda.log"

if [ "$GPUS" -ge 2 ]; then
  torchrun --standalone --nproc_per_node=2 -m st validate --suite distributed --cp 2 \
    --output "$REPORT_DIR/distributed_cp2.json" 2>&1 | tee "$REPORT_DIR/distributed_cp2.log"
fi
if [ "$GPUS" -ge 4 ]; then
  torchrun --standalone --nproc_per_node=4 -m st validate --suite distributed --cp 2 \
    --output "$REPORT_DIR/distributed_cp2_dp2.json" 2>&1 | tee "$REPORT_DIR/distributed_cp2_dp2.log"
  torchrun --standalone --nproc_per_node=4 -m st validate --suite distributed --cp 4 \
    --output "$REPORT_DIR/distributed_cp4.json" 2>&1 | tee "$REPORT_DIR/distributed_cp4.log"
fi
if [ "$GPUS" -ge 8 ]; then
  torchrun --standalone --nproc_per_node=8 -m st validate --suite distributed --cp 8 \
    --output "$REPORT_DIR/distributed_cp8.json" 2>&1 | tee "$REPORT_DIR/distributed_cp8.log"
fi
