# 作业 Delta 与固定门值 Preconditioned DeltaNet

作业原版 Delta 使用固定学习率 0.5。论文对照使用 Tumma、Loo 和 Rus 公开的单头 PyTorch 写入函数，以固定门值运行。两种方法使用相同的键值对和写入顺序，各写一遍，最后冻结矩阵，用完整线索检索全部记忆。[ICML 2026 论文](https://proceedings.mlr.press/v306/tumma26a.html) · [作者代码的固定版本](https://github.com/ntumm120/preconditioned-deltanet/blob/7bd753279af87b39114149a104c5bde9bf67145f/3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py)

## 检索效果与写入内存

| 写入方法 | 平均比特正确率 | 相对作业 Delta | 常驻写入状态 | 写入时 CPU 张量峰值 |
| --- | ---: | ---: | ---: | ---: |
| 作业原版 Delta | 90.83% | 基准 | 8.00 KiB | 16.13 KiB |
| 固定门值 PDN | 91.48% | +0.65 个百分点 | 8.25 KiB | 42.52 KiB |

正确率对 3 种键维数和 5 种记忆数的 15 个条件等权平均，每个条件重复 200 次。逐条件差值使用同一批随机记忆配对计算，图中阴影是每个条件的近似 95% 置信区间。[逐条件正确率与区间](../output/experiment1/preconditioned_delta/accuracy_by_condition.csv) · [正确率图](../output/experiment1/preconditioned_delta/bit_accuracy.png)

表中内存对应 d = 64、内容维数 m = 32、FP32、单序列逐对写入 32 次。PDN 额外保留 64 个 FP32 数，占 0.25 KiB。峰值由 PyTorch CPU 张量分配事件测得，包含临时张量和作业 Delta 的梯度，统计起点在输入预分配之后。[三个维数的内存数据](../output/experiment1/preconditioned_delta/write_memory.csv) · [内存图](../output/experiment1/preconditioned_delta/write_memory.png)

## 为什么这里可以用递减步长解释 PDN

本实验中，每个线索坐标的取值为 ±1/√d，第 t 次写入后每个坐标累积的平方和都等于 t/d。固定门值 PDN 因而对所有坐标使用相同的缩放，实际步长随写入次数递减。用该标量步长独立实现 Delta，和作者递推核对了 4,761,600 个读出比特，符号全部一致。这个解释依赖本实验的等幅线索和固定门值。[运行配置和数值核对](../output/experiment1/preconditioned_delta/run_info.json)

在 d = 64、P = 128 时，改用 PDN 后，最早写入的 32 对记忆正确率从 66.09% 升至 71.18%，最晚写入的 32 对从 90.53% 降至 87.08%。全部 128 对的平均正确率从 76.69% 升至 78.01%。[分写入位置的实测数据](../output/experiment1/preconditioned_delta/accuracy_by_condition.csv)

## 复现实验

在仓库根目录运行任一入口。第一条生成上表所用的 PDN 对照结果，第二条比较作业基线、固定步长 1 的 Delta 及按 d/(d+t−1) 递减的 Delta（t 从 1 开始）。两个入口独立生成结果。

```shell
uv run --locked --extra cpu python -m associative_memory.experiment1.preconditioned
uv run --locked --extra cpu python -m associative_memory.experiment1.optimized_writes
```

步长对照表中 `original` 是原来的 15 个条件，`fresh` 是在两个指定条件各取 100 批新样本的复核。[步长对照表](../output/experiment1/optimized_writes/comparison.csv) · [步长对照图](../output/experiment1/optimized_writes/recall_vs_load.png)

作者参考函数及许可文件位于 [reference/pdn](reference/pdn)。程序在运行前核验源码的 Git blob。
