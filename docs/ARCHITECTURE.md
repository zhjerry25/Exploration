# 执行与语义设计

## 单一代码路径

```text
python -m st / stack-experiment
              │
           st.cli
              ├── st.runtime.engine ── train / eval / plan
              ├── st.tools.validate
              └── st.tools.benchmark

外部 Python ── st.api ── ModelConfig / ExecutionConfig
                         ├── StackModel
                         └── BaselineModel
```

旧 driver 不保留 wrapper。公共 API 不解析命令行，也不初始化分布式组；通信初始化由调用方或 engine 控制。checkpoint 迁移只处理旧结构字段，不重新引入旧训练循环或 EMA。

## 精确 dense gate

令 `a_i = q·k_i / sqrt(D)`，远端块的 `M_j = Σ exp(a_i)`，全部可见远端块的 `Z_R = Σ M_j`。局部窗口保持原定义：当前位置所属块及上一块，裁剪到 causal prefix。远端 token 的 logit 为 `a_i + log M_j − log Z_R`。

因此远端 attention 贡献与 `M_j²` 成比例。前向在线维护局部质量、远端原始质量 `Z_R`、以及加 gate 后的远端质量和 value 加权和；不保存完整 query×context 分数矩阵。反向同时包含 token softmax 的直接导数、块内 `log M_j` 导数和全远端 `−log Z_R` 导数。不能把该算子等同于普通 causal FlashAttention。

Torch 版本以显式分块重算实现一阶梯度，并为双精度 oracle 提供路径。Triton 分开计算 dQ 和 dK/dV，不用跨 program 的浮点原子累加。CUDA fp32 使用 IEEE dot；bf16/fp16 使用相应输入和 fp32 累计，最终容差必须用远程测试确认。

## 共享内存资源适配

kernel 的 shared memory 是单个 thread block 的资源，不是整卡显存。远程 v2
在 FP32 D=128、B=128 时仍需 139264 bytes，而设备上限为 101376 bytes。
矩阵路径要求 N ≥ B；因此仅减少流水级/warp 数不能覆盖这个边界。

矩阵路径仍优先运行。仅当所有配置都在启动前报 `OutOfResources` 时，forward、
dQ、dK/dV、sparse scoring 各自切换到 `st/ops/kernels/streamed.py` 的 Triton 归约路径。
T 从 32 开始，可重试 16/8；T 与模型 B 无关。先跨 T-token tiles 累积完整 block 的
log mass 和加权 value 均值，再应用 gate。remote block 的未归一化质量为 M_j²；
梯度使用完整 block 的 direct-gradient 总和，并保留 −log Z_R 导数。

forward/dQ 每个程序拥有一个 query/head，dK/dV 每个程序拥有不重叠的 key tile/head，
通过重读 block 统计避免梯度原子写。工作集随 T×D 变化，不分配 Q×N 或 Q×blocks
中间矩阵。此路径增加读取/归约成本；矩阵路径能运行时继续使用它。实际性能仍需测量。

两条路径都使用 Triton，FP32 不降精度。编译错误、运行时 CUDA OOM 和数值错误仍然
直接报错。失败的矩阵配置和成功的启动配置会被缓存，避免每次调用重复失败；报告记录
所用 kernel、tile、流水级和编译共享内存用量。验证会强制测试小 tile 前向和全部输入梯度，
包括设备本来能运行矩阵路径的情形。分页 absolute offset 使用运行时参数。

## DP × CP

同一 DP replica 内，输入与 encoder activations 按 token 分片；每次 L 应用交换上一分片的最后一个输入块，包含共享层的每次循环。反向 halo 梯度回到真实拥有者。

G 路径用 sequence/head all-to-all：每个 rank 持有完整序列的部分 heads，分担 dense attention 的算术和内存。raw K/V 在所有 G rounds 之间共享。padding 的 query position 为 -1，尾部键不能被真实 causal query 读取。

DDP 在全部进程间同步共享参数梯度。ZeRO-1 仅分片 optimizer state；参数仍复制，每卡保留完整参数。训练激活约随 CP 数下降，但参数项不随 CP 下降；500M、长序列及大词表组合应先做容量规划。

## 分页 sparse 推理

精确 block scores 必须扫描可见 raw keys。内核只写当前 page 的 block scores，不写完整 token scores。每页候选与累计 top-k 合并，gate 归一化仍包含所有可见远端块；多卡在相同定义下合并候选和质量。

GPU pop 内核直接寻址选中 token，不先构造 `[B,Q,H,K,D]` 的 K/V gather 副本。CPU/磁盘 cache 按页传输，read 仅访问被选块或局部窗口所在页。页大小、score tile、query chunk、词表投影大小分别受预算控制。

局部 RoPE 同时对 query/key 减去 query block 起点，相对旋转的数学定义不变；这也避免 2²⁴ 以上绝对位置被 fp32 整数精度合并。它可能与旧绝对相位实现产生微小舍入差异，不能宣称 checkpoint logits 位级一致。

## 容量与成本边界

raw KV 字节数 = `2 × batch × tokens × dim × dtype_bytes`。例如 dim=1536、bf16、batch=1、32×2²⁰ tokens，仅 raw KV 就需要 192 GiB；8 卡均分后为每卡 24 GiB，尚未包括参数、页缓冲和计算工作区。单卡可使用 host/disk cache，以传输成本换显存。

显存 planner 是保守估算，不能替代实测，也不会捕获 OOM 后静默丢 batch、减少上下文或改变 top-k。精确 query-written scoring 在 Q 个查询下仍为 O(Q×N×D)；全位置 LM 外推仍是二次计算量。磁盘 cache 解决容量问题，不等于任意规模都有可接受的吞吐。
