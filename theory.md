## Stack Model 理论

符号约定：位置 $t$、块 $k=\lfloor t/b \rfloor$、单头（多头逐头成立）、$s_i = q\cdot k_i/\sqrt{d_h}$

## 0. 基础

局部窗口 $L = [(k-1)b,\, t]$，可见块 $B_j$（$j \le k-2$）

**引理 0**：$L \cup \bigcup_{j\le k-2} B_j = [0,t]$，两两不交 $\Rightarrow$ 全注意力配分函数分解为

$$
Z = Z_L + \sum_j M_j
$$

其中块质量

$$
M_j = \sum_{i\in B_j} e^{s_i}
$$

块分数 $s_j = \log M_j$ 即压栈的 logsumexp。

## 1. 全注意力近似

块内归一化输出

$$
\bar v_j = \frac{1}{M_j}\sum_{i\in B_j} e^{s_i} v_i
$$

（“块 $j$ 的局部答案”），则

$$
\mathrm{attn}(t)
= \frac{Z_L\, u_L + \sum_j M_j\, \bar v_j}{Z_L + \sum_j M_j}
$$

压栈归约 $(M_j, \bar v_j)$ 保留了精确混合所需的全部信息。

## 2. 主定理

选中集 $R$（top-$m$），未覆盖质量分数

$$
\varepsilon = \frac{\sum_{j\notin R} M_j}{M}
\qquad \|v_i\| \le V
$$

**定理 1**（gate$=0$ 的干净版）：

$$
\|\mathrm{out} - \mathrm{attn}\|
\le 2V\cdot \varepsilon \cdot \frac{M}{Z_L+M}
\le 2V\varepsilon
$$

证明：$\mathrm{attn} = (A+A_\varepsilon)/(D+D_\varepsilon)$，$\mathrm{out}=A/D$，相减取范数，用 $\|\bar v_j\|\le V$、$\|\mathrm{out}\|\le V$

**推论**：$m=0 \Rightarrow$ 纯局部注意力；$m=G \Rightarrow \varepsilon=0 \Rightarrow$ **等于全注意力**；误差界随 $m$ 单调降。stack 是局部与全注意力之间的可控插值族。

## 3. 选择最优性引理

**引理 2**：按 $s_j=\log M_j$ 排序的 top-$m$ 选择达到覆盖误差下确界

$$
\varepsilon^*_m
= \min_{|R|=m}\frac{\sum_{j\notin R}M_j}{M}
$$

——**注意力的 $m$-尾部质量**。
证明：未覆盖质量由 $m$ 个最大 $M_j$ 最小化。

对照：任何“先压缩再打分”的选择器（NSA/landmark/mean-pool）覆盖误差 $=$ 压缩估计误差 $+$ 尾部误差；此方法恒等于纯尾部误差。$\varepsilon^*_m$ 是学出来的量——harsh gate 的训练压力把它往 $0$ 推（§5）。对于合成任务，选择没有近似误差。

## 4. 生产 gate 的分析

生产 gate $g_j=\mathrm{log\_softmax}(s)_j$ $\Rightarrow$ 保留块被乘

$$
\rho_j=\frac{M_j}{M}\in(0,1]
$$

**命题 3**：

- (i) 质量分布 one-hot 且赢家入选 $\Rightarrow \rho\to 1$，扭曲自动消失；
- (ii) 均匀分布 $\Rightarrow$ 远端整体多压 $\approx 1/G$：**不确定性越高越偏局部**——免费的局部性先验，LM 中大概率有益；
- (iii) 中间态扭曲最大，但此时 §5 的对比梯度正把分布推向 one-hot。

**gate 是自退火扭曲**：早期锐化偏置，margin 起来后读出与全注意力不可区分，与观测一致。

## 5. 门梯度 = 块级 InfoNCE

$$
\frac{\partial g_j}{\partial s_l}=\delta_{jl}-p_l
\;\Longrightarrow\
\frac{\partial \mathcal L}{\partial s_l}
=
\left[\frac{\partial \mathcal L}{\partial g_l}\right]\mathbf 1[l\in R]
-
p_l\sum_{j\in R}\frac{\partial \mathcal L}{\partial g_j}
$$

——**选中块是正样本（推高），全体可见块按 $p_l$ 分摊负样本压力（压低）**，温度 $1$ 的对比学习。这是“选择压力是承重墙”的数学形态：对比目标 $\to$ margin$\uparrow$ $\to$ 质量 one-hot 化 $\to \varepsilon^*\downarrow$ $\to$ gate 退火 $\to$ 逼近全注意力，闭环。relaxed gate 截断正样本推高项 $\to$ margin 停长 $\to$ 大 $G$ 被 max-of-$G$ losers 翻盘——消融零样本归零与此方向一致（`--gate_relaxed` 实验）。

## 6. 分离性论点

不是逼近全注意力（误差 $2V\varepsilon$），而是相反：**全注意力需要随长度增长的 margin，我们不需要**。

- 全注意力中 needle 要压过 $G\cdot b$ 个噪声 token 的**总质量**：所需 margin $\sim \log n$（softmax 稀释，随 $n$ 增长——定量病）。
- top-$m$ 选择只需 needle **排进前 $m$**：所需 margin 取决于噪声次序统计量 $\sim \sqrt{2\log G}$ 量级。

$\log n$ vs $\sqrt{\log G}$：选择是反涣散机制。逼近界（定理 1）降级为安全性陈述（最坏不劣于全注意力减尾部）。此论点与 §5 的对比闭环吻合：对比压力只需把 needle 分数推过低得多的门槛。

## 7. 复杂度陈述

压栈每位置 $n$ 次打分（与全注意力 $QK^\top$ 同阶 FLOPs，但纯归约、只读缓存 $K$、无 $V$ 聚合）；出栈每位置 $m\cdot b+2b$。训练 $O(n^2)$（接受）；推理唯一状态 $=$ token 粒度 KV cache，decode 每步压栈 $O(n)$ 读 $K$（同标准 decode），出栈 $O(m\cdot b)$，无摘要存储与重写。

## 8. 状态与 TODO（诚实清单）

| 条目                                                                                             | 状态                                                              |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| 引理 0 / 定理 1 / 引理 2 / §5 梯度分解                                                          | ✅ 已证（初等）                                                   |
| $\varepsilon^*\to 0$（合成任务收敛后） | ✅ 经验（65k/1M exact$=1.0$）                       |                                                                   |
| gate 自退火（命题 3）                                                                            | ✅ 定性 + 与消融方向一致                                          |
| $\varepsilon^*_m$ 学习动力学（margin 增长速率）                                                | ⬜ 可在 toy 凸设定尝试                                            |
| LM 幂律尾部：$M_{(j)}\propto j^{-\alpha}\Rightarrow \varepsilon^*_m\asymp m^{-(\alpha-1)}$     | ⬜$\alpha$ 从训好的 stlm4096 实测，$\to$ “有效稀疏度”定量图 |
| 多头形式化（界逐头成立，read_out 多$\|W_o\|$ 因子；头的分工 $=$ 不同头把不同块推向 one-hot） | ⬜ 直推但需写                                                     |
| 读侧迭代轮（多跳）的理论                                                                         | ⬜ 等架构实现后                                                   |
