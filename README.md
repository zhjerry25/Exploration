# Stack experiments

统一的 dense 训练 / sparse 推理实验框架。模型和底层算子可单独作为 Python 库使用；所有实验使用同一个命令入口。

```bash
python -m st --help
python -m st train --help
python -m st eval --help
python -m st validate --help
```

训练固定使用精确 dense block-gated attention，`topk` 只控制推理。支持 1–8 个进程的 DP、CP 和组合并行、分块 checkpoint、梯度累积及 ZeRO-1。长上下文推理逐页编码，raw KV 可放在 GPU、CPU 或磁盘；查询状态超过内存预算时也可落盘。

## 开始使用

在远程 NVIDIA 环境中安装与 GPU/驱动兼容的 CUDA PyTorch 和配套 Triton。代码最低接口要求是 Python 3.10、PyTorch 2.5；这不代表任何旧版本的 CUDA wheel 都支持新 GPU。使用环境自带的匹配版本，然后可选择安装本项目：

```bash
python -m pip install --no-deps -e .
```

安装后 `stack-experiment` 与 `python -m st` 调用完全相同的入口。在源码仓库根目录使用后者不需要安装本项目。CPU 可运行 reference 验证；生产执行面向 NVIDIA CUDA。

```bash
# 只规划参数量和显存，不分配完整模型参数，也不要求 GPU
python -m st plan --config configs/stack_500m.json --length 65536

# MQAR dense 训练，2.4M 参数
python -m st train --config configs/stack_2m.json --task mqar \
  --length 512 --npairs 16 --nqueries 16 --batch-size 16 \
  --steps 3000 --lr 5e-4 --eval-every 500 --stop-exact .99 \
  --save runs/mqar512.pt --log runs/mqar512.jsonl

# 多卡 dense LM 训练：无外部数据时用 random 检查吞吐流程
torchrun --standalone --nproc_per_node=8 -m st train \
  --config configs/stack_500m.json --task random --length 65536 \
  --context-parallel 8 --batch-size 1 --grad-accum 4 \
  --steps 20 --log-every 1 --save runs/large-smoke.pt

# 加载旧或新 checkpoint，在 16M 位置范围做 sparse 检索评测
python -m st eval --resume runs/mqar512.pt --task mqar \
  --length 16777216 --npairs 16 --nqueries 16 --batch-size 1 \
  --cache auto --cache-dir /path/to/fast-local-disk/stack-cache \
  --output runs/mqar16m.json
```

`random` 只用于工程测试，不能证明语言建模或检索质量。模型 preset 名称是参数规模标签，准确参数量以 `plan` 输出为准。

## 文档

- [完整命令、数据和迁移说明](docs/COMMANDS.md)
- [公共 Python API](docs/API.md)
- [并行、算子语义和显存设计](docs/ARCHITECTURE.md)
- [验收状态与远程复测](docs/VALIDATION.md)

旧 `st/train.py`、旧 `st/run.py` 和会隐式下载语料的 `st/lmdata.py` 已删除。训练/评测仅由 `st/engine.py` 执行，命令解析仅由 `st/cli.py` 执行。EMA 已移除。旧 checkpoint 的结构和权重仍可加载；历史实验结果文件保留原始记录，不能当作当前实现的验收报告。

## 验证边界

用户回报的并轨前版本通过了 38 项 reference 测试，及前三组 FP32 CUDA 数值用例。随后在 `head_dim=96 / block_size=64` 的 dense kernel 启动时出现共享内存超限。当前版本已修改启动策略，并新增统一入口与 API 回归测试；**修复后的完整 CUDA、多卡、65k 训练和 16M/32M 外推仍需远程验证**。目前没有可据此宣称的整体加速倍数或全规模通过结论。
