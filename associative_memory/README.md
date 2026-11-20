# 可塑性与关联记忆（Homework 3）

公共模型使用 NumPy，命名参考 KAIR 的 ModelBase 接口。

- `models/model_base.py`：仅声明公共接口，保持 KAIR 风格。
- `models/model_memory.py`：实现数据输入、检索、权重快照、重置和 Delta 更新。
- `ModelHebbian`：继承 `ModelDelta`，只重写 `optimize_parameters()`。
- `ModelZero`：不写入的零矩阵基线。

在仓库根目录的 Python 环境中：

```python
import numpy as np
from associative_memory.models import ModelDelta

model = ModelDelta(d=64, m=32, eta=0.5)
rng = np.random.default_rng(42)
k = rng.choice([-1.0, 1.0], size=64) / np.sqrt(64)
v = rng.choice([-1.0, 1.0], size=32)
model.feed_data({"k": k, "v": v})
model.optimize_parameters()
model.feed_data({"q": k})
model.test()
results = model.current_results()
```

`z` 是连续输出，用于计算 MSE；`v_hat` 是符号输出，用于计算比特正确率。
零输出对应 +1。训练每次只接收一条配对；检索支持单条或批量查询。
模型不自动归一化输入、不裁剪或衰减权重。

多轮训练、随机顺序、遮挡和回放由后续实验脚本控制。评估指标和数据生成也放在模型外。
实验三使用 `state_dict()` 和 `load_state_dict()` 复制 A 训练完成后的权重。
快照不共享数组内存；`reset()` 清零权重并清空当前输入与结果。

已完成公共模型和实验一；实验二、三的正式结果输出尚未实现。

## 单元测试

```powershell
uv run --locked --extra cpu pytest associative_memory/tests -q
```

Delta 与固定版本 FLA 官方 Gated DeltaNet 参考函数比较每步读出和最终权重：关闭遗忘（g=0），beta=eta，scale=1，状态转置为 W。覆盖两组随机种子、三种学习率和两种维数，并从非零权重开始。Hebbian 使用批量闭式解交叉验证；另测试零基线、快照、重置和检索不修改权重。参考源码及 MIT 许可证位于 tests/reference。

## 实验一：容量

```powershell
uv run --locked python -m associative_memory.experiment1
# 单次快速查看
uv run --locked python -m associative_memory.experiment1 --seeds 42
```

默认20个独立种子，m=32，eta=0.5，遍历作业规定的15种(d,P)组合；每条配对只写入一次。每个组合独立生成数据，三种方法共享配对和写入顺序。

默认输出到 associative_memory/output/experiment1：逐次指标 metrics.csv、均值与样本标准差 summary.csv、参数 config.json，以及 accuracy.png、mse.png、weight_norm.png。正确率以 P/d 为横轴，MSE 和权重范数以 P 为横轴；阴影是跨种子的样本标准差，不是置信区间。

讨论时关注：Hebbian 检索含目标项与其他配对的串扰，增加 d 可减小独立随机线索的重叠，但权重存储增至 m*d 个标量（本实现 float64，每个8字节）。Delta 修正预测误差，两者权重范数的大小不能直接代替检索质量。零矩阵基线连续 MSE 恒为1，随机目标上的比特正确率在0.5附近波动。

## 随机数管理

所有正式实验通过 random.py 的 make_rng(seed, experiment, purpose, *coordinates) 创建随机流。相同身份可复现，不依赖调用顺序；不同用途独立。固定使用 PCG64，名称通过 SHA256 稳定编码，不使用 Python hash。

实验一分开 data 和 order；实验二可用 masks，实验三可用 buffer 和 replay。方法间公平比较时先生成共享数据或顺序，再传给模型。相同参数每次调用会从随机流起点重建，应保留返回的 rng 持续抽样。随机方案变更后实验一结果已重新生成，旧结果不再逐项一致。
