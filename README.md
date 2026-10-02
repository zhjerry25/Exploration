# Stack experiments

Dense 训练、sparse 推理的实验框架。**所有命令都从 `python -m st` 开始。**
Baseline Transformer 使用 PyTorch 原生 SDPA 自动选择 FlashAttention；Stack 保持精确
gated dense 训练和分页 sparse 推理。评测统一输出 loss/bpc/ppl/accuracy/exact，并可按
绝对位置分桶或只监督尾部 query。
在仓库根目录运行；不需要知道内部 Python 文件的位置。

| 想做什么                | 命令                                                 |
| ----------------------- | ---------------------------------------------------- |
| 看有哪些命令            | `python -m st --help`                              |
| 看模型大小和显存估算    | `python -m st plan --config configs/stack_2m.json` |
| 训练                    | `python -m st train --help`                        |
| 加载 checkpoint 评测    | `python -m st eval --help`                         |
| 验证正确性              | `python -m st validate --help`                     |
| 测 attention 速度和显存 | `python -m st benchmark --help`                    |

## 环境

远程机器需要支持该 NVIDIA GPU 的 CUDA PyTorch 与配套 Triton；接口最低要求为
Python 3.10、PyTorch 2.5，但这些最低版本不保证支持新款 GPU。使用已匹配的环境。
可选安装 `python -m pip install --no-deps -e .`，随后也能使用 `stack-experiment` 命令。

## 先验证，再开始实验

```bash
python -m st validate --suite reference --output runs/validation/reference-v3.json
# 快速复测此前 D=128/B=128 的 shared-memory 失败及小 tile 的前后向
python -m st validate --suite resources --output runs/validation/resources-v3.json
# 完整单卡 CUDA 矩阵
python -m st validate --suite cuda --output runs/validation/cuda-v3.json
```

多卡验证：`GPUS=2 bash scripts/remote_validate.sh`，将 2 改为实际可用的 1/2/4/8。
验证通过后，`GPUS=2 bash scripts/remote_smoke.sh` 检查训练、保存和重载流程。

## 训练与评测

```bash
# 单卡 MQAR；topk 不改变训练，训练始终 dense
python -m st train --config configs/stack_2m.json --task mqar \
  --length 512 --npairs 16 --nqueries 16 --batch-size 16 \
  --steps 3000 --eval-every 500 --save runs/mqar.pt --log runs/mqar.jsonl

# 加载权重做 sparse 外推；自动选择 GPU/CPU/磁盘缓存
python -m st eval --resume runs/mqar.pt --task mqar \
  --length 1048576 --npairs 16 --nqueries 16 --batch-size 1 \
  --cache auto --cache-dir runs/cache --output runs/mqar-1m.json
```

## Benchmark 怎么跑

**不需要 checkpoint 或训练数据，算子 benchmark 用固定随机输入。** 在一张 GPU 上运行：

```bash
# 一条命令比较 eager、bounded torch、Triton 的 dense 前向+反向
python -m st benchmark --compare --length 512

# 单独测 dense 前向+反向；默认 queries=length
python -m st benchmark --operation dense --length 4096 --precision bf16

# 单独测长上下文 sparse 前向；只查询 16 个位置
python -m st benchmark --operation sparse --length 1048576 --queries 16 --topk 64

# 对此前失败的 FP32 配置单独计时
python -m st benchmark --operation dense --length 385 --queries 5 \
  --head-dim 128 --block-size 128 --precision fp32
```

终端显示毫秒延迟和显存，完整报告自动保存到 `runs/benchmarks/`；`--output PATH`
可改位置。报告包含实际 kernel 路径与共享内存用量。预热不计入延迟。
`--compare` 从小长度开始，eager 会物化 Q×N 分数。

**完整模型训练吞吐**用下面的命令，读日志中的 `tokens_per_second`：

```bash
torchrun --standalone --nproc_per_node=2 -m st train \
  --config configs/stack_2m.json --task random --length 8192 \
  --context-parallel 2 --batch-size 1 --steps 20 --log-every 5 --save '' \
  > runs/benchmarks/train-cp2.jsonl 2>&1
```

The shell redirection keeps `--log` from being interpreted as a `torchrun`
`--log-dir` abbreviation on PyTorch versions that parse launcher options after
`-m st`.

`random` 只测工程性能，不证明任务质量；算子毫秒数也不代表完整训练吞吐。

## 目录

```text
configs/              Stack 2M / 100M / 500M 与公平 baseline 2M 预设
st/
  api.py, config.py   外部接口：build_model / load_model / 配置
  cli.py              统一命令路由
  models/             StackModel、baseline、局部 blocks
  ops/                dense/sparse 算子，kernels/ 为 Triton 实现
  runtime/            训练、并行、checkpoint、分页推理、容量规划
  data/               合成任务与 mmap token 文件
  tools/              validate 与 benchmark
scripts/              远程验收；experiments/ 为长时实验配方
tests/                正确性回归测试
docs/                 详细说明；research/ 保存历史研究记录
data/, runs/          本地语料与输出，不纳入版本控制
```

- [完整命令、并行训练、数据与续训](docs/COMMANDS.md)
- [公共 Python API](docs/API.md)
- [算法、并行与资源设计](docs/ARCHITECTURE.md)
- [Camera-ready 优化记录与验收](docs/research/Optimization.md)
- [已验证结果与尚待验收的项目](docs/VALIDATION.md)

`from st import StackModel, build_model, load_model, InferenceSession` 等公共接口保持不变。
旧 checkpoint 仍可加载，旧 Python 模块导入通过别名转到唯一实现。
