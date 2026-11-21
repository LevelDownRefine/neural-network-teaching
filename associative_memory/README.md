# 可塑性与关联记忆（Homework 3）

模型使用原生 PyTorch 接口，输入为 FP32 Tensor。环境使用仓库的 `cpu` 依赖组（`uv sync --locked --extra cpu`）。

- `ModelMemory`：零初始化的可训练参数 W，`forward(**kwargs)`（读取 q） 返回 `q @ W.T`，支持单条或批量查询。
- `delta_loss` / `hebbian_loss`：逐对训练的目标函数，配合普通 SGD 实现两种塑性规则。

```python
from copy import deepcopy
import torch
from associative_memory.models import ModelMemory
from associative_memory.losses import delta_loss, hebbian_loss

model = ModelMemory(d=64, m=32)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
criterion = delta_loss  # 换成 hebbian_loss 即为 Hebbian
rng = torch.Generator().manual_seed(123)
k = (2 * torch.randint(2, (64,), generator=rng) - 1).float() / 8
v = (2 * torch.randint(2, (32,), generator=rng) - 1).float()

optimizer.zero_grad()
criterion(model(q=k), v).backward()
optimizer.step()

model.eval()
with torch.no_grad():
    z = model(q=k)
    v_hat = torch.where(z >= 0, 1, -1)
snapshot = deepcopy(model.state_dict())
model.load_state_dict(snapshot)
```

Delta 目标为 `0.5 * (pred - target).square().sum()`，得到 `W += lr * outer(v - W @ k, k)`。
Hebbian 目标为 `-(pred * target).sum()`，得到 `W += lr * outer(v, k)`；该目标无有限最小值，仅用于产生规则对应的梯度，评估仍使用正确率和 MSE。
严格对应作业时逐对更新，不使用 momentum、weight decay 或 mean reduction。Zero 基线使用同一个零初始化模型，不执行训练。

不再使用 feed_data、optimize_parameters、test、current_results 或 reset。训练轮数、顺序、设备搬运、归一化和评估由调用方控制。重新开始可创建新模型。需要独立快照时必须 deepcopy，原生 state_dict 的张量共享模型存储。若使用带动量的优化器，恢复训练还应保存和恢复 optimizer.state_dict()。

## 数据与测试

`data.py` 的 `generate_pairs(rng, p, d, m)` 返回 K、V 和随机写入顺序。随机数生成器由实验持有，各方法共享数据与顺序；正式实验脚本另行提交。

```powershell
uv run --locked --extra cpu pytest associative_memory/tests -q
```

测试保留固定版本 FLA 官方参考函数对照（源码和 MIT 许可证位于 tests/reference），比较 Delta 每步读出和最终权重；另验证两种 SGD 更新与显式公式一致、Hebbian 闭式解、原生状态恢复及模块组合，以及容量、遮挡和回放流程。

调用使用 `model(**batch)`，读取 `q`，忽略其余字段；缺少 `q` 会抛出 KeyError。ModelMemory 直接继承 nn.Module，保留维度、FP32 和有限值检查。

## 公共训练循环

`train.py` 提供 `train_epoch(trainer, keys, values, order)`，按 order 逐对更新一轮（允许重复索引用于回放）。输入使用同设备 Tensor。函数不生成随机数、不创建 optimizer、不负责评估或保存；跨轮使用同一个 optimizer，保留其状态。Zero 基线不调用训练函数。

```python
from associative_memory.train import Trainer, train_epoch
trainer = Trainer(model, criterion, optimizer)
train_epoch(trainer, keys, values, order)
```

NumPy 回归测试使用提交 `960150d` 的原始实现转为 FP32 作对照（`tests/reference/numpy_memory.py`），固定 seed=123，覆盖容量、遮挡和回放。检查每步 W、连续读出和符号差异。NumPy 与 PyTorch 即使同为 FP32 也不保证逐位相同；近零读出的符号可能因舍入变化，测试会报告这类差异，不要求正确率完全相等。

`Trainer` 组合 model、criterion 和 optimizer；`step(inputs, target)` 通过 `model(**inputs)` 计算，执行清梯度、反向传播和更新，返回脱离计算图的更新前损失。推理仍直接调用 model；快照分别使用 model 和 optimizer 的原生接口。
