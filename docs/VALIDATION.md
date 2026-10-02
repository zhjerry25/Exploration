# 验收状态

## 已收到的用户回报

回报对应提交 `e5220dc` 的并轨前实现：

- 38 项 reference 测试通过，6.510 秒。
- FP32 dense 前后向及 sparse 前向在 `(N, block, head_dim)` = `(17,2,16)`、`(129,16,32)`、`(257,32,64)` 通过。
- 上述 dense 前向最大绝对误差不超过 `8.35e-7`，输入梯度不超过 `9.08e-7`；sparse 前向不超过 `5.37e-7`。
- 接下来的 `(193,64,96)` dense 前向因 shared memory 需求 143424 bytes 超过 101376-byte 硬件限制而无法启动。
- bf16 矩阵、后续更大 head_dim、多卡和性能结果未随此次回报提供。

这些是用户提供的测试结果，不是本地执行结果。没有把旧 reference 通过视为当前并轨版本、全 GPU 矩阵或所有目标规模已经通过。

## CUDA v2 新回报

用户回报：FP32 `D=128 / block_size=128` 的 `_forward` 在全部启动配置重试后仍失败：
共享内存需求 **139264 bytes > 101376 bytes**。这证明上一轮仅调流水级和 tile 的修复
不充分；该报错不是整卡显存耗尽，也没有提供 GPU 硬件故障的证据。
没有收到完整 cuda-v2 JSON，因此不推断其余用例已经通过。

## 当前修改与最小复测

增加完整 block 跨小 tile 归约的 Triton forward/dQ/dK/dV/scoring 路径，保持 gate 语义。
矩阵路径资源不足时独立切换；编译错误、CUDA OOM 仍失败。目录按 models/ops/runtime/data/tools
分组，命令改用子命令专属帮助，benchmark 新增同输入比较与默认报告路径。

同步整个工作区（包括新增文件、目录移动和删除）后，在仓库根目录运行：

```bash
python -m st validate --suite reference --output runs/validation/reference-v3.json
python -m st validate --suite resources --output runs/validation/resources-v3.json
python -m st validate --suite cuda --output runs/validation/cuda-v3.json
```

`resources` 专门复测 D128/B128、D256/B128、D96/B64，包含自动选择、小 tile 强制执行、
前向/全部输入梯度、非零页面 offset 的 scoring 和 sparse 输出。完整 `cuda` 还覆盖短 block、
dummy queries、尾部与端到端 bf16。新测试代码不等于通过记录；当前无远程 v3 结果。

不必清空 Triton cache。若失败，请保留 `case_start`、完整 traceback、报告的 `environment`
与 `kernel_launches`。本地仅进行静态检查；数值、实际 JIT 编译、资源与性能由远程验证。

## 后续验收矩阵

当前代码还增加了以下本地可重复检查：

- `MetricAccumulator` 的流式 loss/bpc/ppl、exact 与位置桶边界。
- Baseline 的 `math` SDPA 与整层 checkpoint 前后向梯度一致性。
- `--supervision tail --tail-tokens K` 与 `--metric-buckets` 的统一 CLI 路径。
- CUDA KV tier 下 query state 驻留 GPU 的分页推理路径（需远端 CUDA 才能覆盖）。

```bash
GPUS=2 bash scripts/remote_validate.sh
GPUS=2 bash scripts/remote_smoke.sh
```

将 GPUS 改为实际可用的 1/2/4/8。完整矩阵依次包含：

1. 数学、因果性、梯度、checkpoint、流式 cache 与公共 API 的 reference tests。
2. CUDA fp32/bf16：dense forward/dQ/dK/dV、sparse forward、共享 G、完整模型 loss 和参数梯度。
3. CP2/CP4/CP8 及 CP2×DP2：StackModel/baseline 参数梯度对照、尾部 padding、零监督 rank、不同 mask、跨分片 sparse prefill。
4. 实际 driver：梯度累积、ZeRO-1、checkpoint、重新加载、训练与推理入口。
5. 2.4M/101M/约498M 参数、512/8k/65k dense 训练的端到端吞吐和显存。
6. 已训练权重在 1M/4M/16M/32M 的检索正确率、耗时及 GPU/host/disk 容量，覆盖单卡和可用的多卡配置。

最后两项尚无通过记录；命令示例不等于实际规模验收。不同 GPU、词表和 batch 组合也不能靠一个小样本通过推断全部完成。

## 测量约定

`benchmark` 使用 CUDA events、预热、重复采样，并报告 allocated/reserved 峰值。`train` 输出端到端 optimizer-step 吞吐；不同 DP/CP 比较时固定全局有效 batch 和监督定义。`random` 用于性能，不能用于验证模型任务质量。所有结果需连同 PyTorch/Triton/CUDA/GPU 信息保存。
