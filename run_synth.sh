#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# S 组：合成任务极限外推（passkey + MQAR，单种子，无 baseline）
#
# 模型固定：stack, arch=Lx2,G, d=256, 2.43M 参数。
# 保底路径（topic.md 已验证）：MQAR 128 点火 → 512 续训；passkey 512 点火。
# 极限：零样本评估阶梯 4096 → 65536 → 1M → 2M → 4M，逐档落盘，互不依赖。
#
# MQAR 配置（自行定夺）：npairs=16 固定 + nqueries=16 + 单 token 键。
#   评估长度增长时键值密度自然稀释（4M 时 16 对 ≈ 4ppm），即 Zoology 意义上
#   更难的稀疏设置；值为 10 分类数字，难度全部压在检索侧。
#
# 资源核算（Pro 6000 96GB，fp32 评估保精度）：4M 时单条 eval 峰值 ~13GB
#   (x 4.3GB + raw K/V 4.3GB + logits 2GB + ts ~1GB)，时间大头是线性编码
#   (~6 TFLOP/条序列)，4M 档约 20-30min，全程约 2-3h。无 OOM 风险。
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p runs
say() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

# ── S1 MQAR：128 点火 → 512 续训（保底，历史复现路径）
say "S1 MQAR 128 点火"
python -m st.train --task mqar --n 128 --npairs 16 --nqueries 16 \
  --steps 3000 --bs 64 --lr 1e-3 --bf16 --stop_exact 0.99 --save runs/s1_mqar128.pt
say "S1 MQAR 512 续训（保底 checkpoint）"
python -m st.train --task mqar --n 512 --npairs 16 --nqueries 16 \
  --steps 2000 --bs 64 --lr 5e-4 --bf16 --stop_exact 0.99 \
  --resume_weights_only runs/s1_mqar128.pt --save runs/s1_mqar512.pt

say "S1 MQAR 零样本外推阶梯"
python -m st.train --task mqar --n 512     --npairs 16 --nqueries 16 --eval_only --resume runs/s1_mqar512.pt --bs 16
python -m st.train --task mqar --n 4096    --npairs 16 --nqueries 16 --eval_only --resume runs/s1_mqar512.pt --bs 8
python -m st.train --task mqar --n 65536   --npairs 16 --nqueries 16 --eval_only --resume runs/s1_mqar512.pt --bs 2
python -m st.train --task mqar --n 1000000 --npairs 16 --nqueries 16 --eval_only --resume runs/s1_mqar512.pt --bs 1
python -m st.train --task mqar --n 2000000 --npairs 16 --nqueries 16 --eval_only --resume runs/s1_mqar512.pt --bs 1
python -m st.train --task mqar --n 4000000 --npairs 16 --nqueries 16 --eval_only --resume runs/s1_mqar512.pt --bs 1

# ── S2 passkey：512 点火（保底）→ 同一外推阶梯
say "S2 passkey 512 点火"
python -m st.train --task passkey --n 512 --steps 1500 --bs 64 --lr 5e-4 \
  --bf16 --stop_exact 0.99 --save runs/s2_passkey512.pt

say "S2 passkey 零样本外推阶梯"
python -m st.train --task passkey --n 512     --eval_only --resume runs/s2_passkey512.pt --bs 16
python -m st.train --task passkey --n 4096    --eval_only --resume runs/s2_passkey512.pt --bs 8
python -m st.train --task passkey --n 65536   --eval_only --resume runs/s2_passkey512.pt --bs 4
python -m st.train --task passkey --n 1000000 --eval_only --resume runs/s2_passkey512.pt --bs 1
python -m st.train --task passkey --n 2000000 --eval_only --resume runs/s2_passkey512.pt --bs 1
python -m st.train --task passkey --n 4000000 --eval_only --resume runs/s2_passkey512.pt --bs 1

say "S 组全部结束"
