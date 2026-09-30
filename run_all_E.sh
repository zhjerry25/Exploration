#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# E 组实验总脚本（train dense infer sparse）
#
# 原则：训练稠密（read_m >= n/b）
#      稀疏 top-m 选择只在零样本外推 eval 时启用
#
# ─────────────────────────────────────────────────────────────────────────────

# 该评测已完成

mkdir -p runs
say() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

# extrapolate eval：$1=checkpoint $2=task $3...=length list
zs() { CK=$1; TASK=$2; shift 2
  for N in "$@"; do
    python -m st.train --model stack --task $TASK --n $N --eval_only --resume $CK --bs 2 ${EXTRA}
  done
}

# ── E1 dense LM@512 → zero-shot extrapolate 4k/16k/65k

say "E1 stack dense LM@512"

python -m st.train --model stack    --task lm --n 512 --arch Lx2,G --bs 32 --steps 30000 --lr 5e-4 --bf16 --save runs/e1_stack.pt

say "E1 baseline LM@512"

python -m st.train --model baseline --task lm --n 512 --layers 3  --bs 32 --steps 30000 --lr 5e-4 --bf16 --save runs/e1_base.pt

say "E1 zero-shot extrapolation"

EXTRA= zs runs/e1_stack.pt lm 4096 16384 65536
for N in 4096 16384 65536; do
  python -m st.train --model baseline --task lm --n $N --eval_only --resume runs/e1_base.pt --bs 2
done

say "E1 baseline"

python -m st.train --model baseline --task passkey --n 512 --steps 1500 --bs 64 --lr 5e-4 --bf16
python -m st.train --model baseline --task mqar --n 128 --npairs 16 --nqueries 16 --steps 3000 --bs 64 --lr 1e-3 --bf16

# ── E2 G/L scaling scan（dense；LM@512，8k steps）

say "E2 local scaling"
for A in "L,G" "Lx2,G" "Lx4,G" "Lx8,G" "(L)x4,G"; do
  CK="runs/e2_w_${A//[^A-Za-z0-9]/_}.pt"
  python -m st.train --model stack --task lm --n 512 --arch "$A" --bs 32 --steps 8000 --lr 5e-4 --bf16 --save $CK
  EXTRA= zs $CK lm 4096 65536
done

say "E2 Global cycling（including independent params Gx4）"

for A in "Lx2,G" "Lx2,(G)x2" "Lx2,(G)x4" "Lx2,(G)x8" "Lx2,Gx4"; do
  CK="runs/e2_r_${A//[^A-Za-z0-9]/_}.pt"
  python -m st.train --model stack --task lm --n 512 --arch "$A" --bs 32 --steps 8000 --lr 5e-4 --bf16 --save $CK
  EXTRA= zs $CK lm 4096 65536
done

# ── E3 block size scan

say "E3 block size scan"

for B in 8 16 32 64; do
  CK="runs/e3_b$B.pt"
  python -m st.train --model stack --task lm --n 512 --b $B --read_m 64 --arch Lx2,G --bs 32 --steps 8000 --lr 5e-4 --bf16 --save $CK
  EXTRA= zs $CK lm 4096 65536
done

# ── E4 harder MQAR（key values increase with n，2048 pairs when 65k；dense igition + extrapolation）

say "E4 harder MQAR"

python -m st.train --model stack --task mqar --n 512 --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 \
  --steps 3000 --bs 64 --lr 1e-3 --bf16 --stop_exact 0.99 --save runs/e4_stack.pt
python -m st.train --model baseline --task mqar --n 512 --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 \
  --layers 3 --steps 3000 --bs 64 --lr 1e-3 --bf16 --save runs/e4_base.pt
EXTRA="--npairs_density 0.03125 --nkeytoks 2 --nqueries 16" zs runs/e4_stack.pt mqar 4096 16384 65536
for N in 4096 16384 65536; do
  python -m st.train --model baseline --task mqar --n $N --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 --eval_only --resume runs/e4_base.pt --bs 2
done

# ── E5 training length scan（b=32；n=256..2048 all dense）

say "E5 training length scan（b=32）"

for N in 256 512 1024 2048; do
  CK="runs/e5_n$N.pt"
  python -m st.train --model stack --task lm --n $N --b 32 --read_m 64 --arch Lx2,G --bs 32 --steps 8000 --lr 5e-4 --bf16 --save $CK
  EXTRA= zs $CK lm 4096 16384 65536
done

say "E group experiments end"

# Optional shutdown
/usr/bin/shutdown
