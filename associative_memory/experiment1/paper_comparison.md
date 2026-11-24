# 论文方法对照

在同一批随机键值上比较固定步长 Delta、经典递归最小二乘（RLS）和 Preconditioned DeltaNet（PDN）的作者参考写入函数。这里运行单头、固定门值的递推，直接测量一次写入后对原键的检索表现。RLS 采用岭正则 0.01，原版 Delta 的学习率为 0.5。

## 运行

在仓库根目录执行以下命令生成 PDN、Delta 与 RLS 对照。两个入口各自计算所需方法，不依赖另一个入口的 CSV。

```shell
uv run --locked --extra cpu python -m associative_memory.experiment1.preconditioned
```

如只需要固定步长 Delta、递减步长 Delta 和 RLS 的比较，可以单独运行下列命令。

```shell
uv run --locked --extra cpu python -m associative_memory.experiment1.optimized_writes
```

## 如何读结果

结果保存在 `associative_memory/output/experiment1/`。

| 文件 | 内容 |
| --- | --- |
| `optimized_writes/comparison.csv`、`recall_vs_load.png` | 不同步长及 RLS 与 Hebbian、Delta、零矩阵的检索表现 |
| `preconditioned_delta/accuracy_by_condition.csv`、`bit_accuracy.png` | 三种写入规则的比特正确率、配对差值与逐条件 95% 区间 |
| `preconditioned_delta/write_memory.csv`、`write_memory.png` | 各方法跨写入保留的状态字节数，以及单序列逐条写入 32 对时的 CPU 张量分配峰值 |
| `preconditioned_delta/run_info.json` | 软件版本、采样种子、门值、上游源码版本和独立数值检查 |

每个键由等幅的随机正负分量构成并归一化。键维数取 32、64、128，记忆数取 8、16、32、64、128，内容维数为 32。三个方法以相同顺序逐对写入一次，写完后用最终状态检索。每个组合独立重复 200 次，差值按同一批样本配对计算。

PDN 在此配置下关闭遗忘，写入步长参数为 0.5，预条件器对每条键等权累积。由于每个键的各坐标平方相同，对角预条件器产生统一的步长缩放。程序用独立的标量步长实现核验读出。此结果适用于上述配对任务和固定门值配置。

作者参考函数及两份许可文件保存在 `reference/pdn/`。运行前程序核验该函数的 Git blob。 [作者固定版本源码](https://github.com/ntumm120/preconditioned-deltanet/blob/7bd753279af87b39114149a104c5bde9bf67145f/3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py)
