# 公共 Python API

外部实验代码使用 `st` 包的公开导出，无需依赖命令解析器、内置合成任务或私有 Triton kernel。

```python
from st import (
    StackModel, BaselineModel,
    ModelConfig, ExecutionConfig,
    build_model, load_model,
    InferenceSession, ParallelContext, TokenDataset,
    MetricAccumulator, tail_mask,
)
```

## 构建与加载

```python
import torch
from st import ModelConfig, ExecutionConfig, build_model, load_model

config = ModelConfig(
    model="stack", vocab_size=32000, dim=1024, heads=16,
    block_size=16, arch="Lx7,G", topk=64,
)
execution = ExecutionConfig(
    backend="auto", checkpoint_chunks=True,
    encoder_chunk=1024, query_chunk=128, loss_chunk=128,
    flash_attention="auto", attention_q_chunk=128,
    attention_kv_chunk=4096,
)
model = build_model(config, execution=execution, device="cuda")

# 配置可转换为普通 JSON-compatible 字典
structure = config.to_dict()
restored_config = ModelConfig.from_dict(structure)

# 新/旧 checkpoint 均从保存的结构构建；返回 eval 模型
loaded = load_model(
    "runs/model.pt", device="cuda", dtype=torch.bfloat16,
    execution=execution, topk=32,
)
```

`build_model` 返回原生 `torch.nn.Module`，参数保持标准 `state_dict` 格式。embedding 与 lm_head 的权重共享、`(L)xN/(G)xN` 的共享权重保持不变。`ExecutionConfig` 仅控制执行，不改变架构参数。`load_model` 只恢复权重；完整训练续训使用统一 CLI。

baseline 使用 `ModelConfig(model="baseline", layers=3, ...)`，同样接受执行配置。其
attention 通过 PyTorch SDPA 自动选择 FlashAttention/memory-efficient kernel；
`flash_attention="flash"` 可在验收时强制 fused kernel，`"math"` 仅用于数值对照。
它不能使用 `InferenceSession` 的 sparse raw-KV 缓存。

## 训练：直接传入自己的 batch

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
input_ids = torch.randint(32000, (2, 512), device="cuda")
targets = torch.randint(32000, (2, 512), device="cuda")
mask = torch.ones_like(targets, dtype=torch.bool)

optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    result = model(input_ids, targets=targets, sup=mask)
    loss = result["loss_sum"] / result["count"]
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

输入/target 均为 `[B,N]`，`sup` 为 bool `[B,N]`。`targets == -100` 永远不计入 loss，允许每行监督数量不同或某些 rank 没有监督 token。整个有效 batch 必须至少有一个监督 token。StackModel 的此入口始终 dense，与 `model.topk` 无关。

共同返回值：

| 键 | 内容 |
|---|---|
| `loss_sum` | 当前 rank 有效 token 的 CE 总和，保留反向图 |
| `count` | 当前 rank 有效 token 数 |
| `correct` | 正确预测的有效 token 数 |
| `row_errors` | 每行错误 token 数，可用于跨 CP 归并 exact accuracy |

StackModel 还返回 packed `positions`/`hits`；它们不是 baseline 的共同接口。分块 CE 不返回完整 `[B,N,V]` logits。调用方负责对 loss 做期望的归一化。

统一评测可用 `MetricAccumulator` 流式累加 query chunks，结果同时包含 `loss`、`bpc`、
`ppl`、token `accuracy`、row-level `exact` 和按绝对位置划分的 `buckets`。CLI 的
`--supervision tail --tail-tokens K` 为训练与评测提供尾部监督；`--metric-buckets
0,1024,4096,...` 自定义桶边界，默认自动生成四个等宽桶。

原始小规模接口仍可用：`model(ids, sup=mask)` 返回 `[B,N,V]`，非监督位置为零；`compact=True` 返回等长监督位置的 `[B,S,V]`。此接口保留完整编码状态和 logits，适合 reference/小规模研究。长上下文使用下面的 session API。

## 自定义多卡训练

`ParallelContext.initialize(cp_size=N)` 读取 torchrun 环境并创建通信组。`parallel.shard` 将 CPU batch 按 block 对齐分片，最后一个分片用 padding 补齐；模型输入的 `offset` 是全局序列位置。

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from st import ParallelContext

parallel = ParallelContext.initialize(cp_size=2, device="cuda")
device = torch.device("cuda", torch.cuda.current_device())
model.to(device)
wrapped = DistributedDataParallel(model, device_ids=[device.index],
                                  broadcast_buffers=False)

# ids_cpu / targets_cpu / mask_cpu: CP peers 的同一个 CPU batch；
# 不同 DP replica 应使用不同数据 RNG stream。
ids, offset = parallel.shard(ids_cpu, model.block_size)
labels, _ = parallel.shard(targets_cpu, model.block_size, -100)
mask, _ = parallel.shard(mask_cpu, model.block_size, False)
with torch.autocast("cuda", dtype=torch.bfloat16):
    result = wrapped(ids.to(device), targets=labels.to(device),
                     sup=mask.to(device), parallel=parallel, offset=offset)
    count = result["count"].detach().clone()
    dist.all_reduce(count)
    loss = result["loss_sum"] * parallel.world_size / count
loss.backward()
```

乘 `world_size` 是为了抵消 DDP 随后的平均。所有 rank 必须执行相同的通信次序，包括没有本地监督 token 的 rank。通常直接用 CLI 可避免自己实现数据流、累积、归一化和保存同步。

## 超长推理：分页与迭代输出

```python
import torch
from st import load_model, InferenceSession

model = load_model("runs/mqar512.pt", device="cuda", dtype=torch.bfloat16)
# 输入保留在 CPU；此处应换成自己的完整 prompt。
ids_cpu = torch.randint(128, (1, 1_000_000))
positions_cpu = torch.tensor([[999_995, 999_996, 999_997, 999_998, 999_999]])

with InferenceSession(
    model, cache="auto", cache_dir="/data/stack-cache",
    encoder_chunk=4096, page_tokens=65536, query_chunk=16,
    workspace_mb=256,
) as session:
    session.prefill(ids_cpu, positions_cpu)
    for requested_slice, logits, selection in session.iter_logits():
        prediction = logits.argmax(-1)
        # requested_slice 对应 positions_cpu 的列，而不是原 prompt 区间。
        consume(requested_slice, prediction.cpu())
```

`prefill` 的 query positions 为 CPU `[B,Q]`；可乱序、可重复，每行 Q 相同。只保留请求位置的 encoded query states。超过 RAM query-state 预算时需要 `cache_dir`；状态落盘后仍逐 chunk 读取。`return_selection=True` 可取得最后一个 G round 的 selection（完整 query chunk 的 indices/scores/log_zr）；通常指标只需 logits，不必返回 selection。

`consume` 表示调用方自己的写盘/累计指标逻辑。不要把长 LM 的全部 logits 再 `cat` 回内存。session 是一个 prompt 的 write-once prefill/read 接口，不是可 append 的自回归 decode cache；新 prompt 使用新 session。

多卡推理时，将 `group=dist.group.WORLD` 传给 session，并确保所有 rank 使用相同 ids/positions/模型权重。每个 rank 编码和保存自己的 KV 分片；top-k、gate 归一化和 attention 输出在全局合并。结束时关闭 session；context manager 会清理磁盘临时文件。

## 数据与底层算子

`TokenDataset(path, dtype="uint16", start=0, stop=None).batch(batch_size, length, generator)` 返回 CPU `(input_ids, targets, mask, None)`。`start/stop` 是 token offset；没有网络访问或隐式语料下载。

研究自定义 attention 时，可从 `st.ops.attention` 导入 `dense_attention`，从 `st.ops.sparse` 导入 `sparse_attention`/`TensorPages`。这些是低层接口；布局是 `[B,Q,H,D]` 和 `[B,N,H,D]`，positions 是 `[B,Q]`，`-1` 表示 dummy query。dense 算子提供一阶梯度，不支持二阶导数。Triton 支持 fp16/bf16/fp32、每头维度不超过 256、block_size 为 2–128 的 2 的幂；其他配置使用 torch reference/fallback，明确指定 `backend="triton"` 时不支持的配置会报错。

目录重组不改变上述顶层 API。旧 `st.blocks`、`st.stack_model`、`st.baseline`、`st.attention`、`st.sparse`、`st.parallel`、`st.inference`、`st.checkpoint`、`st.memory`、`st.token_data` 模块导入保留别名，指向同一实现。底层 Triton 文件属于内部接口，现位于 `st/ops/kernels/`。
