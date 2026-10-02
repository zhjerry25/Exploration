# Camera-ready optimization record

本次优化保持 `e5220dc` 之前已经定义的 Stack 语义：局部 L 写入 raw K/V，G 轮次使用
query-written block mass，训练走精确 gated dense attention，推理走 exact score/top-k
分页读取。性能改动集中在执行层，所以旧 checkpoint 的参数名和数学函数不变。

## 训练路径

Stack 的 Triton dense kernel 仍由 `backend=auto` 自动选择；启动资源不足时会按设备限制
选择 matrix 或 streamed reduction kernel，FP32 不偷偷降为 TF32。portable torch 路径的
query/KV tiles 现在由 `ExecutionConfig.attention_q_chunk` 和
`attention_kv_chunk` 控制，默认 128/4096，长上下文仍不保存 Q×N score matrix。

Baseline 是参数量和层数匹配的普通 causal Transformer。它通过无显式 mask 的
`torch.nn.functional.scaled_dot_product_attention` 进入 PyTorch 的 FlashAttention 或
memory-efficient SDPA；只有 `flash_attention="math"` 才强制数学参考路径。Baseline 的
checkpoint 现在包住整层 QKV、SDPA 和 FFN，避免只重算 FFN 却把注意力中间量留在显存。

训练引擎在 CUDA 上自动使用 fused AdamW；启用 ZeRO-1 时使用 foreach AdamW 以控制额外
workspace，旧版 PyTorch 会回退到兼容实现。DP×CP 通信和有效 token 归一化保持不变。

## 推理路径

`InferenceSession` 仍按 encoder chunk 写入 KV page。选择 CUDA KV tier 时，查询状态也
驻留 GPU，避免每个 query chunk 的 host-to-device 往返；CPU/disk tier 保持原来的容量
边界和清理语义。`torch.inference_mode()` 消除了推理阶段的 autograd bookkeeping。

显存规划仍按每 rank 的 raw KV、page、score tile 和 query chunk 分开估计。对于 16M
上下文，若 raw KV 无法放入 GPU，`cache=auto` 会选择 CPU 或 disk；这解决容量问题但不
改变 exact scoring 的 O(QND) 计算量。训练 CLI 仍拒绝超过 65k 的 dense length，避免
把推理能力误当成训练能力。

## 结果与验收

评测不再把长序列 logits 拼回内存。`MetricAccumulator` 对每个 query chunk 流式累加
loss、bpc、ppl、token accuracy、row exact，并支持绝对位置桶；`--supervision tail`
可以固定尾部监督，便于检索任务与 LM 的公平比较。

本地参考套件覆盖数学、因果性、梯度、checkpoint、分页 cache、配置和公共 API；远端
CUDA 验收使用：

```bash
GPUS=2 bash scripts/remote_validate.sh
GPUS=2 bash scripts/remote_smoke.sh
bash scripts/benchmark_camera_ready.sh
```

`benchmark_camera_ready.sh` 固定随机任务、长度、有效 batch 和步骤，分别记录 Stack 与
Baseline 的 optimizer-step tokens/s，再记录 dense operator 的 eager/torch/Triton 对照。
报告必须同时保存 GPU、PyTorch、CUDA、Triton、SDPA 能力和显存峰值；首次 Triton 编译
窗口不参与稳定吞吐比较。短基准默认使用 `CHECKPOINT=0`，确保两种模型在同一重算
开关下比较；长上下文或大模型可用 `CHECKPOINT=1`。在 AutoDL 的 2x RTX 4090 上，
长度 8192、bf16、全局 batch 2、24 steps 的稳定窗口测得 Stack 自动 query tile
约 166--178k tokens/s，Baseline Flash SDPA 约 136--152k tokens/s；Stack/程式峰值
显存约 0.35/0.20 GiB。旧的 128-query 固定分块只有约 20--25k tokens/s，瓶颈是
Python/Triton 调度而非算术吞吐。
