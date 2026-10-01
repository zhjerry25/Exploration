# 统一命令参考

唯一推荐入口：`python -m st <command>`。可安装的 `stack-experiment` 是相同入口。训练和推理不会因不同命令走两套驱动。

| 命令 | 用途 |
|---|---|
| `plan` | 参数计数和不同 CP 规模的显存估算；使用 meta 模型 |
| `train` | dense loss、反向传播、优化器、周期评测、保存/续训 |
| `eval` | checkpoint 评测；StackModel 使用分页 sparse 路径 |
| `validate` | reference、CUDA 数值一致性、多卡梯度与推理验收 |
| `benchmark` | CUDA attention 算子时间、峰值显存 |

## 配置与优先级

JSON 的根节点只有 `model` 和 `runtime`，键名用下划线，例如 `batch_size`；命令参数用连字符，例如 `--batch-size`。参考 `configs/stack_2m.json`、`stack_100m.json`、`stack_500m.json`。

新建实验：命令显式选项覆盖 JSON，JSON 覆盖默认值。加载 checkpoint：模型种类、维度、heads、block_size、arch/layers、词表等结构由 checkpoint 决定；`--length`、执行设置和 StackModel 的 `--topk` 可调整。`--topk` 不改变 dense 训练。

| 主要参数 | 含义 |
|---|---|
| `--model stack / baseline` | StackModel 或标准 dense causal baseline |
| `--dim`, `--heads`, `--vocab-size` | 模型大小；RoPE 要求每头维度为偶数 |
| `--arch 'Lx2,(G)x4'` | L/G 应用顺序，括号表示共享权重；L 必须在 G 前 |
| `--layers` | baseline 层数 |
| `--length` | 输入长度，训练上限 65536，外推可更长 |
| `--batch-size` | 每个 DP replica 的 microbatch，不是每张 CP GPU 的独立 batch |
| `--grad-accum` | 累积 microbatch 数 |
| `--precision bf16 / fp32` | 训练 bf16 autocast / fp32；独立推理使用相应权重与缓存 dtype |
| `--backend auto / triton / torch` | auto 对不支持的布局用 bounded torch 路径；Triton 资源不足时切换小 tile Triton，编译/数值错误不会被吞掉 |
| `--context-parallel auto / N` | N 必须整除进程数和 heads；auto 是启发式规划，不是性能调优结果 |
| `--[no-]checkpoint-chunks` | 局部计算和 vocabulary head 的激活重计算 |
| `--[no-]optimizer-shard` | 多卡 ZeRO-1；默认启用 |
| `--activation-offload` | 将 autograd 保存张量移到 pinned host memory，消耗 RAM/PCIe 带宽 |
| `--encoder-chunk`, `--query-chunk`, `--loss-chunk` | 编码 token、每 rank 训练 query、loss head 的分块大小 |
| `--memory-fraction` | 显存预算比例；默认 .8 |
| `--allow-over-budget` | 明确越过保守训练估算；不会保证实际运行不 OOM |

全局有效 batch = `batch_size × (WORLD_SIZE / CP) × grad_accum`。用 `--context-parallel 1` 选择纯 DP。最多 8 个进程；各 CP rank 处理相同样本的不同序列分片，各 DP replica 使用不同数据 RNG 流。

## 单卡、DP、CP

```bash
python -m st train --config configs/stack_2m.json --task passkey \
  --length 512 --batch-size 16 --steps 3000 --save runs/passkey.pt

torchrun --standalone --nproc_per_node=4 -m st train \
  --config configs/stack_2m.json --task mqar --length 512 \
  --context-parallel 1 --batch-size 8 --steps 3000 --save runs/mqar-dp.pt

# CP=4、DP=2；每个样本跨 4 张 GPU，8 张 GPU 同时处理 2 组样本
torchrun --standalone --nproc_per_node=8 -m st train \
  --config configs/stack_100m.json --task tokens --tokens /data/train.bin \
  --token-dtype uint16 --vocab-size 32000 --length 65536 \
  --context-parallel 4 --batch-size 1 --grad-accum 4 \
  --steps 1000 --save runs/lm.pt
```

增大词表会改变参数量；先执行同参数的 `plan`。baseline 使用同一训练引擎及 CP all-to-all，但其全局层始终是普通 dense causal attention，不存在 sparse 推理或 raw-KV 分页加速。baseline 独立评测应使用单进程；多进程目前会重复评测同一数据，不提供推理加速。

## 数据

`passkey`、`copying`、`mqar` 保留原生成规则。MQAR 选项为 `--npairs`、`--nqueries`、`--nkeytoks 1/2`、`--npairs-density`。密度选项大于零时，pair 数随 length 增长；不能超过 key 字母表的组合数。

`tokens` 读取扁平、无文件头的本机字节序 token ID 文件，支持 `uint8/uint16/int32/int64`。使用 mmap，按 batch 切片后才转 int64。词表大小由 `--vocab-size` 指定，文件中的 ID 必须落在词表范围内。周期评测必须提供独立的 `--validation-tokens`。

```bash
python -m st train --task tokens --tokens /data/train.bin \
  --validation-tokens /data/val.bin --token-dtype uint16 --vocab-size 32000 \
  --length 4096 --batch-size 1 --eval-every 100 --eval-batches 4 \
  --steps 1000 --save runs/tokens.pt
```

`enwik8` 使用已经解压的 100,000,000-byte 文件；默认路径 `data/enwik8`，可用 `--tokens` 指定。训练使用前 90M bytes，验证/测试各 5M bytes，词表固定 256。不自动下载、不自动解压。独立评测的 `--split` 默认 `val`。

```bash
python -m st train --task enwik8 --tokens /data/enwik8 --length 4096 \
  --batch-size 4 --steps 1000 --eval-every 100 --save runs/enwik8.pt
python -m st eval --task enwik8 --tokens /data/enwik8 --split test \
  --resume runs/enwik8.pt --length 65536 --batch-size 1
```

## checkpoint、续训和 early stop

`--save` 默认 `runs/stack.pt`；实验间应使用不同路径。传 `--save ''` 可关闭保存。`--save-every` 按 optimizer step 计数，最后一步自动保存。checkpoint 以临时文件加原子 rename 写入，保留权重共享、optimizer、step、每 rank RNG 和拓扑。

```bash
# 从长任务的中间 checkpoint 继续；保留原实验参数及总 steps
python -m st train --task mqar --length 512 --batch-size 16 \
  --npairs 16 --nqueries 16 --steps 3000 --lr 5e-4 \
  --resume runs/mqar512.pt --save runs/mqar512.pt

# 更换长度/拓扑/训练目标：只加载权重，重置 optimizer、schedule 和 RNG
python -m st train --task mqar --length 4096 --batch-size 2 \
  --steps 2000 --lr 1e-4 --resume runs/mqar512.pt --weights-only \
  --save runs/mqar4096.pt
```

`--steps` 是总目标步数。完整 resume 要求保存时的任务、长度、batch、累积数、精度、优化参数与拓扑一致；旧 checkpoint 没有数据 RNG 时会明确提示无法精确延续数据流。

使用 `--eval-every 500 --eval-batches 4 --stop-exact .99` 启用周期评测与 early stop。周期评测使用当前 fp32 master weights，不会把训练参数永久改成 bf16；独立 `eval --precision bf16` 会使用 bf16 权重。比较实验时需记录这种数值精度区别。周期评测时间不计入训练吞吐窗口。

## 超长 sparse 外推

```bash
torchrun --standalone --nproc_per_node=8 -m st eval \
  --resume runs/mqar512.pt --task mqar --length 33554432 \
  --npairs 16 --nqueries 16 --batch-size 1 --topk 64 \
  --cache auto --cache-dir /data/stack-cache \
  --page-tokens 65536 --inference-query-chunk 16 --workspace-mb 256 \
  --output runs/mqar32m.json
```

StackModel 独立评测时，所有 rank 协同处理同一 prompt 的不同 KV 分片。它不要求推理 GPU 数整除 heads。`--context-parallel` 是训练参数，推理按实际 WORLD_SIZE 分片。

| 推理设置 | 含义 |
|---|---|
| `--cache auto/cuda/cpu/disk` | raw-KV 存储层；auto 依据当前可用显存、RAM 及是否提供磁盘目录选择 |
| `--cache-dir` | 磁盘 KV/query-state 的临时文件目录；session 退出后清理 |
| `--page-tokens` | 缓存传输页大小；planner 可缩小到 workspace 允许的范围 |
| `--inference-query-chunk` | 同时扫描上下文的 query 数 |
| `--workspace-mb` | 页、打分和 head 工作区的规划预算 |
| `--max-query-states-mb` | query states 的 RAM 预算，超过后需要磁盘目录 |
| `--eval-positions K` | 等距选 K 个查询位置；0 表示任务全部监督位置 |

合成检索任务默认只评价答案位置。LM 默认全位置；在 16M/32M 上对每个 token 进行精确 query-written block scoring，计算量仍然是二次增长。仅研究长上下文上的少数查询时应显式设置 `--eval-positions`，并在结果中区分抽样和完整 LM 评测。

## Benchmark：算子性能与模型性能

算子 benchmark 不加载模型/checkpoint，不读取语料，只测随机 Q/K/V 上的 attention。
用一张 NVIDIA GPU，直接运行 `python -m st benchmark`。不要对这个子命令使用多进程
`torchrun`；多卡模型性能用下面的 `train --task random`。

```bash
# 512 长度，bf16，同输入比较三个 dense 实现
python -m st benchmark --compare --length 512
# sparse 比较仅包括 torch 和 triton
python -m st benchmark --compare --operation sparse --length 4096 --queries 16
# 大模型头维度和 128-token block 的 FP32 资源回归性能
python -m st benchmark --length 385 --queries 5 --head-dim 128 \
  --block-size 128 --precision fp32 --iterations 20
```

| 参数 | 默认值与含义 |
|---|---|
| `--operation dense/sparse` | dense：前向+反向；sparse：前向（scoring、top-k、pop） |
| `--backend triton/torch/eager` | 默认 triton；eager 仅适合小规模 dense 对照 |
| `--compare` | 固定种子、尺寸和精度，顺序测试可用的对照实现 |
| `--length` | KV token 数，默认 512 |
| `--queries` | dense 默认全部位置；sparse 默认最后最多 16 个位置 |
| `--batch-size`, `--heads`, `--head-dim` | 默认 1、4、64；head-dim 是每头维度，不是总 dim |
| `--block-size`, `--topk` | 默认 16、64；topk 只控制 sparse |
| `--precision bf16/fp32` | 默认 bf16，FP32 不启用 TF32 |
| `--page-tokens` | sparse KV 页大小，默认 65536；输入 KV 仍整体驻 GPU |
| `--warmup`, `--iterations` | 默认 3、10；先预热，再用 CUDA events 计时 |
| `--output` | 默认 `runs/benchmarks/<operation>-<backend-or-compare>.json` |

终端显示延迟与显存。JSON 保存各次采样、环境、参数、allocated/reserved 峰值、
实际 kernel 路径和共享内存字节数。`--compare` 顺序执行，不自动跳过失败；如果
资源或精度不支持，命令会报错。eager 会分配 Q×N，因此先比较 512，再逐步放大。

小 tile 路径主要保证资源兼容，会增加重复读取和计算，速度以报告为准。
算子 benchmark 的 KV 在 GPU 内存中；要测 CPU/磁盘分页开销，用 `eval --resume ...`
的 `seconds` 和显存记录，不能用算子结果代替分页推理结果。

完整模型的 optimizer-step 吞吐：

```bash
torchrun --standalone --nproc_per_node=2 -m st train \
  --config configs/stack_2m.json --task random --length 8192 \
  --context-parallel 2 --batch-size 1 --steps 20 --log-every 5 --save '' \
  --log runs/benchmarks/train-cp2.jsonl
```

看 JSONL 的 `tokens_per_second`、`peak_allocated_gib`。首次编译会影响首个窗口，
比较后续稳定窗口。比较 DP/CP 时固定全局有效 batch。该命令不评估学习质量。

## 正确性验证

```bash
python -m st validate --suite reference --output runs/validation/reference.json
python -m st validate --suite resources --output runs/validation/resources.json
python -m st validate --suite cuda --output runs/validation/cuda.json
torchrun --standalone --nproc_per_node=2 -m st validate \
  --suite distributed --cp 2 --output runs/validation/cp2.json

python -m st benchmark --operation dense --backend triton --length 4096 \
  --output runs/bench-dense.json
python -m st benchmark --operation sparse --backend triton --length 1048576 \
  --queries 16 --output runs/bench-sparse.json
```

`GPUS=1/2/4/8 bash scripts/remote_validate.sh` 运行对应矩阵；随后运行 `GPUS=N bash scripts/remote_smoke.sh` 检查真实训练、optimizer、checkpoint、加载和算子计时。算子 benchmark 不代表完整模型吞吐。

## 从旧命令迁移

| 旧项 | 新项 |
|---|---|
| `python -m st.train ...` | `python -m st train ...` |
| `python -m st.run train/eval/plan` | `python -m st train/eval/plan` |
| `--eval_only` | `eval` 子命令 |
| `--n / --b / --d / --bs` | `--length / --block-size / --dim / --batch-size` |
| `--read_m` | `--topk`；不再需要随训练长度上调 |
| `--bf16` | `--precision bf16`；新驱动默认 bf16 |
| `--resume_weights_only PATH` | `--resume PATH --weights-only` |
| `--eval_every / --ckpt_every` | `--eval-every / --save-every` |
| `--stop_exact / --npairs_density` | `--stop-exact / --npairs-density` |
| `--task lm` | `--task enwik8`，或通用 `--task tokens` |
| `--tag` | 显式 `--log runs/NAME.jsonl` 和 `--save runs/NAME.pt` |
| `--prof / --selftest` | `benchmark` / `validate` 子命令；JSONL 含端到端训练吞吐 |
| `--ema` | 已删除，不读写或应用 EMA |

旧 flags 不保留静默兼容层，错误拼写会直接失败。旧参数权重/checkpoint 兼容层保留在 `st.runtime.checkpoint`。实验配方移至 `scripts/experiments/synthetic.sh`、`scripts/experiments/enwik8_and_scaling.sh`；它们仍是长时实验配方，请先通过小规模验收再启动。
