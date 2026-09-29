#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# S2 组：合成任务收官（3 seeds 卫生 + 硬件内极限 + 加难 MQAR 终审）
#
# A. seeds 1,2 完整阶梯（seed 0 已完成全 1.0）：MQAR 128→512→4M；passkey 512→4M
#    协议与 seed 0 实测一致：无 early-stop，足步数点火。
# B. 极限档（seed 0）：8M / 16M 零样本。Pro 6000 96GB 内存内可达
#    （16M/bs1 峰值 ~45GB）；32M+ 需要推理 kernel，属于下一阶段，不做。
#    G=1M 块时的 sel_cov/gate_margin/needle_block_hit = √(2logG) 预言检验。
# C. 加难 MQAR 终审：双 token 键 + 密度 1/32，15k 步最后一次点火尝试。
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p runs
say() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

# ── A. 3 seeds（seed 0 已归档，补 1/2）
for S in 1 2; do
  say "A MQAR seed=$S 点火与续训"
  python -m st.train --task mqar --n 128 --npairs 16 --nqueries 16 \
    --steps 6000 --bs 64 --lr 1e-3 --bf16 --seed $S --save runs/s2_mqar128_s$S.pt
  python -m st.train --task mqar --n 512 --npairs 16 --nqueries 16 \
    --steps 4000 --bs 64 --lr 5e-4 --bf16 --seed $S \
    --resume_weights_only runs/s2_mqar128_s$S.pt --save runs/s2_mqar512_s$S.pt
  say "A MQAR seed=$S 外推阶梯"
  for N in 512 4096 65536 1000000 2000000 4000000; do
    python -m st.train --task mqar --n $N --npairs 16 --nqueries 16 \
      --eval_only --resume runs/s2_mqar512_s$S.pt --bs 1 --seed $S
  done
  say "A passkey seed=$S 点火与外推阶梯"
  python -m st.train --task passkey --n 512 --steps 3000 --bs 64 --lr 5e-4 \
    --bf16 --seed $S --save runs/s2_passkey512_s$S.pt
  for N in 512 4096 65536 1000000 2000000 4000000; do
    python -m st.train --task passkey --n $N \
      --eval_only --resume runs/s2_passkey512_s$S.pt --bs 1 --seed $S
  done
done

# ── B. 极限档（seed 0 的既有 checkpoint；每档独立落盘，失败不传染）
say "B 极限外推：MQAR 8M / 16M"
for N in 8000000 16000000; do
  python -m st.train --task mqar --n $N --npairs 16 --nqueries 16 \
    --eval_only --resume runs/s1_mqar512.pt --bs 1
done
say "B 极限外推：passkey 8M / 16M"
for N in 8000000 16000000; do
  python -m st.train --task passkey --n $N \
    --eval_only --resume runs/s2_passkey512.pt --bs 1
done

# ── C. 加难 MQAR 终审（双 token 键，密度 1/32，15k 步，无 early-stop）
say "C 加难 MQAR 终审点火"
python -m st.train --task mqar --n 512 --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 \
  --steps 15000 --bs 64 --lr 1e-3 --bf16 --save runs/s2_mqarhard.pt
say "C 加难 MQAR 零样本（无论点火成败都记录）"
for N in 4096 65536; do
  python -m st.train --task mqar --n $N --npairs_density 0.03125 --nkeytoks 2 --nqueries 16 \
    --eval_only --resume runs/s2_mqarhard.pt --bs 2
done

say "S2 组全部结束"
