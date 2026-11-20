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

目前完成公共模型；三个核心实验流程及图表尚未实现。

## 单元测试

```powershell
uv run --locked --extra cpu pytest associative_memory/tests -q
```

Delta 与固定版本 FLA 官方 Gated DeltaNet 参考函数比较每步读出和最终权重：关闭遗忘（g=0），beta=eta，scale=1，状态转置为 W。覆盖两组随机种子、三种学习率和两种维数，并从非零权重开始。Hebbian 使用批量闭式解交叉验证；另测试零基线、快照、重置和检索不修改权重。参考源码及 MIT 许可证位于 tests/reference。
