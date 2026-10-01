#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# ─────────────────────────────────────────────────────────────────────────────
# E 组实验总脚本（train dense infer sparse）
#
# 原则：训练始终稠密；topk 仅控制推理。
#      稀疏 top-m 选择只在零样本外推 eval 时启用
#
# ─────────────────────────────────────────────────────────────────────────────

# 该评测已完成

mkdir -p runs
say() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

# extrapolate eval：$1=checkpoint $2=task $3...=length list
zs() { CK=$1; TASK=$2; shift 2
  for N in "$@"; do
    python -m st eval --model stack --task $TASK --length $N --resume $CK --batch-size 2 ${EXTRA:-} --precision fp32 --eval-batches 8
  done
}

# ── E1 dense LM@512 → zero-shot extrapolate 4k/16k/65k

say "E1 stack dense LM@512"

python -m st train --model stack    --task enwik8 --length 512 --arch Lx2,G --batch-size 32 --steps 30000 --lr 5e-4 --precision bf16 --save runs/e1_stack.pt --eval-every 500 --eval-batches 4

say "E1 baseline LM@512"

python -m st train --model baseline --task enwik8 --length 512 --layers 3  --batch-size 32 --steps 30000 --lr 5e-4 --precision bf16 --save runs/e1_base.pt --eval-every 500 --eval-batches 4

say "E1 zero-shot extrapolation"

EXTRA= zs runs/e1_stack.pt enwik8 4096 16384 65536
for N in 4096 16384 65536; do
  python -m st eval --model baseline --task enwik8 --length $N --resume runs/e1_base.pt --batch-size 2 --precision fp32 --eval-batches 8
done

say "E1 baseline"

python -m st train --model baseline --task passkey --save runs/e1_base_passkey.pt --length 512 --steps 1500 --batch-size 64 --lr 5e-4 --precision bf16 --eval-every 500 --eval-batches 4
python -m st train --model baseline --task mqar --save runs/e1_base_mqar.pt --length 128 --npairs 16 --nqueries 16 --steps 3000 --batch-size 64 --lr 1e-3 --precision bf16 --eval-every 500 --eval-batches 4

# ── E2 G/L scaling scan（dense；LM@512，8k steps）

say "E2 local scaling"
for A in "L,G" "Lx2,G" "Lx4,G" "Lx8,G" "(L)x4,G"; do
  CK="runs/e2_w_${A//[^A-Za-z0-9]/_}.pt"
  python -m st train --model stack --task enwik8 --length 512 --arch "$A" --batch-size 32 --steps 8000 --lr 5e-4 --precision bf16 --save $CK --eval-every 500 --eval-batches 4
  EXTRA= zs $CK enwik8 4096 65536
done

say "E2 Global cycling（including independent params Gx4）"

for A in "Lx2,G" "Lx2,(G)x2" "Lx2,(G)x4" "Lx2,(G)x8" "Lx2,Gx4"; do
  CK="runs/e2_r_${A//[^A-Za-z0-9]/_}.pt"
  python -m st train --model stack --task enwik8 --length 512 --arch "$A" --batch-size 32 --steps 8000 --lr 5e-4 --precision bf16 --save $CK --eval-every 500 --eval-batches 4
  EXTRA= zs $CK enwik8 4096 65536
done

# ── E3 block size scan

say "E3 block size scan"

for B in 8 16 32 64; do
  CK="runs/e3_b$B.pt"
  python -m st train --model stack --task enwik8 --length 512 --block-size $B --topk 64 --arch Lx2,G --batch-size 32 --steps 8000 --lr 5e-4 --precision bf16 --save $CK --eval-every 500 --eval-batches 4
  EXTRA= zs $CK enwik8 4096 65536
done

# ── E4 harder MQAR（key values increase with n，2048 pairs when 65k；dense igition + extrapolation）

say "E4 harder MQAR"

python -m st train --model stack --task mqar --length 512 --npairs-density 0.03125 --nkeytoks 2 --nqueries 16 \
  --steps 3000 --batch-size 64 --lr 1e-3 --precision bf16 --eval-every 500 --stop-exact 0.99 --save runs/e4_stack.pt --eval-batches 4
python -m st train --model baseline --task mqar --length 512 --npairs-density 0.03125 --nkeytoks 2 --nqueries 16 \
  --layers 3 --steps 3000 --batch-size 64 --lr 1e-3 --precision bf16 --save runs/e4_base.pt --eval-every 500 --eval-batches 4
EXTRA="--npairs-density 0.03125 --nkeytoks 2 --nqueries 16" zs runs/e4_stack.pt mqar 4096 16384 65536
for N in 4096 16384 65536; do
  python -m st eval --model baseline --task mqar --length $N --npairs-density 0.03125 --nkeytoks 2 --nqueries 16 --resume runs/e4_base.pt --batch-size 2 --precision fp32 --eval-batches 8
done

# ── E5 training length scan（b=32；n=256..2048 all dense）

say "E5 training length scan（b=32）"

for N in 256 512 1024 2048; do
  CK="runs/e5_n$N.pt"
  python -m st train --model stack --task enwik8 --length $N --block-size 32 --topk 64 --arch Lx2,G --batch-size 32 --steps 8000 --lr 5e-4 --precision bf16 --save $CK --eval-every 500 --eval-batches 4
  EXTRA= zs $CK enwik8 4096 16384 65536
done

say "E group experiments end"

# No automatic shutdown.
