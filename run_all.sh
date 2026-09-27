#!/bin/bash

# 创建保存目录，避免因目录不存在而报错
mkdir -p runs

# ── E1 基线对比（同参数同层数；旗舰图 = posloss_bpc_curve）
python -m st.train --model stack    --task lm --n 4096 --arch Lx2,G --bs 16 --steps 15000 --lr 5e-4 --bf16 --save runs/e1_stack.pt
python -m st.train --model baseline --task lm --n 4096 --layers 3  --bs 16 --steps 15000 --lr 5e-4 --bf16 --save runs/e1_base.pt
for N in 16384 65536; do
  python -m st.train --model stack    --task lm --n $N --eval_only --resume runs/e1_stack.pt --bs 2
  python -m st.train --model baseline --task lm --n $N --eval_only --resume runs/e1_base.pt  --bs 2
done
python -m st.train --model baseline --task passkey --n 512 --steps 1500 --bs 64 --lr 5e-4 --bf16
python -m st.train --model baseline --task mqar --n 128 --npairs 16 --nqueries 16 --steps 3000 --bs 64 --lr 1e-3 --bf16

# ── E2 双侧 scaling 线扫（LM@4096，8k 步/点位）
for A in "L,G" "Lx2,G" "Lx4,G" "Lx8,G" "(L)x4,G"; do
  python -m st.train --model stack --task lm --n 4096 --arch "$A" --bs 16 --steps 8000 --lr 5e-4 --bf16 --save runs/e2_w_${A//[^A-Za-z0-9]/_}.pt
done
for A in "Lx2,G" "Lx2,(G)x2" "Lx2,(G)x4" "Lx2,(G)x8" "Lx2,Gx4"; do
  python -m st.train --model stack --task lm --n 4096 --arch "$A" --bs 16 --steps 8000 --lr 5e-4 --bf16 --save runs/e2_r_${A//[^A-Za-z0-9]/_}.pt
done

# ── E3 block size 扫描（固定 m·b=1024）
for BM in "8 128" "16 64" "32 32" "64 16"; do
  set -- $BM
  python -m st.train --model stack --task lm --n 4096 --b $1 --read_m $2 --arch Lx2,G --bs 16 --steps 8000 --lr 5e-4 --bf16 --save runs/e3_b$1.pt
done

# ── E4 加难 MQAR（键值随 n 增长，65k 时 2048 对）
python -m st.train --model stack --task mqar --n 512 --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 \
  --steps 3000 --bs 64 --lr 1e-3 --bf16 --stop_exact 0.99 --save runs/e4_stack.pt
python -m st.train --model baseline --task mqar --n 512 --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 \
  --layers 3 --steps 3000 --bs 64 --lr 1e-3 --bf16 --save runs/e4_base.pt
for N in 4096 16384 65536; do
  python -m st.train --model stack    --task mqar --n $N --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 --eval_only --resume runs/e4_stack.pt --bs 2
  python -m st.train --model baseline --task mqar --n $N --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 --eval_only --resume runs/e4_base.pt  --bs 2
done

# 全部实验结束后关机
# 注意：shutdown 通常需要 root 权限，如果当前用户无权限请改用 sudo 或提前切换 root
/usr/bin/shutdown