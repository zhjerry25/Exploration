# 实验纪律（摘自原 Multigrid Attention 项目 map.md）

> **2026-09-23 注记**：主线已从 MGA 切换到 **stack model**（见 topic.md 与 theory.md）。原 map.md 的 MGA 实验集合（E1-E3、基线矩阵）为历史路线，不在本仓库；以下保留通用的严谨性纪律。

## 严谨性纪律

1. **主表 3 seeds**（mean±std）；探索性图可单 seed
2. 检索任务判据统一 **≥99% solve**（exact-match），不足者进故障分析
3. 所有对比同 token 量、同优化器配方；FLOPs 与 tok/s 同时报（x 轴双轨）
4. 基线自实现均须通过 sanity（同任务上对 full 的已知结果复核 + 泄漏测试）
5. 数据切分固定（enwik8 90/5/5），种子固定，jsonl 全量归档
