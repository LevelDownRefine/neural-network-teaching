# 可塑性与关联记忆（Homework 3）

公共模型使用 NumPy，命名参考 KAIR 的 ModelBase 接口。

- `models/model_base.py`：数据输入、检索、权重快照及保存加载。
- `ModelHebbian`、`ModelDelta`：只重写 `optimize_parameters()`。
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
快照不共享数组内存；磁盘 `save()` 另外保存学习率，`load()` 恢复它。

目前完成公共模型；三个核心实验流程及图表尚未实现。
