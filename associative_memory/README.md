# 可塑性与关联记忆（Homework 3）

模型使用原生 PyTorch 接口，输入为 FP32 Tensor。环境使用仓库的 `cpu` 依赖组（`uv sync --locked --extra cpu`）。

- `ModelMemory`：零初始化的可训练参数 W，`forward(**kwargs)`（读取 q） 返回 `q @ W.T`，支持单条或批量查询。
- `DeltaTrainer.loss` / `HebbianTrainer.loss`：逐对训练的目标函数，配合普通 SGD 实现两种塑性规则。

```python
from copy import deepcopy
import torch
from associative_memory.models import ModelMemory
from associative_memory.train import DeltaTrainer, HebbianTrainer

model = ModelMemory(d=64, m=32)
trainer = DeltaTrainer(model, lr=0.5)  # 或 HebbianTrainer
rng = torch.Generator().manual_seed(123)
k = (2 * torch.randint(2, (64,), generator=rng) - 1).float() / 8
v = (2 * torch.randint(2, (32,), generator=rng) - 1).float()

trainer.step({"q": k}, v)

model.eval()
with torch.no_grad():
    z = model(q=k)
    v_hat = torch.where(z >= 0, 1, -1)
snapshot = deepcopy(model.state_dict())
model.load_state_dict(snapshot)
```

Delta 目标为 `0.5 * (pred - target).square().sum()`，得到 `W += lr * outer(v - W @ k, k)`。
Hebbian 目标为 `-(pred * target).sum()`，得到 `W += lr * outer(v, k)`；该目标无有限最小值，仅用于产生规则对应的梯度，评估仍使用正确率和 MSE。
严格对应作业时逐对更新，不使用 momentum、weight decay 或 mean reduction。Zero 基线使用同一个零初始化模型，ZeroTrainer 不写入权重。

不再使用 feed_data、optimize_parameters、test、current_results 或 reset。训练轮数、顺序、设备搬运、归一化和评估由调用方控制。重新开始可创建新模型。需要独立快照时必须 deepcopy，原生 state_dict 的张量共享模型存储。若使用带动量的优化器，恢复训练还应保存和恢复 optimizer.state_dict()。

## 数据与测试

`data.py` 的 `generate_pairs(rng, p, d, m)` 返回 K、V 和随机写入顺序。实验一沿用 NumPy 基线的随机流，使各方法和两种实现可以逐比特对照。

### 实验一

在仓库根目录运行主实验。每个设置使用 200 批相同的随机记忆，各条配对只写入一次，写完再用完整线索检索。结果写入 `associative_memory/output/experiment1/`。

```shell
uv run --locked --extra cpu python -m associative_memory.experiment1.run
```

`experiment1_curves.png` 包含比特正确率随记忆负载的曲线，以及连续读出 MSE 和权重范数随记忆数量的曲线。`experiment1_summary.csv` 保存每组的均值和标准差。`experiment1_comparison.csv` 和同名 PNG 对照 PyTorch FP32 与归档的 NumPy FP64 结果。

下列模块分别复算独立 NumPy 基线、比较递减步长 Delta 与 RLS、比较官方 PDN 写入递推与两种对照。拓展模块会先生成缺少的对照数据。

```shell
uv run --locked --extra cpu python -m associative_memory.experiment1.numpy_baseline
uv run --locked --extra cpu python -m associative_memory.experiment1.optimized_writes
uv run --locked --extra cpu python -m associative_memory.experiment1.preconditioned
```

PDN 使用单头和固定门值，直接读写作业给定的键值。写入步长为 0.5，无遗忘，预条件器对每条键等权累积。RLS 使用岭正则 0.01。`preconditioned_delta/` 保存正确率、配对差值以及持续写入状态的字节数。CPU 张量峰值由 PyTorch profiler 在单批、每次输入一条配对的情况下测量。程序会核验官方参考函数的 Git blob，随附许可证位于 `experiment1/reference/pdn/`。[作者参考实现](https://github.com/ntumm120/preconditioned-deltanet/blob/7bd753279af87b39114149a104c5bde9bf67145f/3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py)

### 测试

```powershell
uv run --locked --extra cpu pytest associative_memory/tests -q
```

测试保留固定版本 FLA 官方参考函数对照（源码和 MIT 许可证位于 tests/reference），比较 Delta 每步读出和最终权重；另验证两种 SGD 更新与显式公式一致、Hebbian 闭式解、原生状态恢复及模块组合，以及容量、遮挡和回放流程。

调用使用 `model(**batch)`，读取 `q`，忽略其余字段；缺少 `q` 会抛出 KeyError。ModelMemory 直接继承 nn.Module，保留维度、FP32 和有限值检查。

## 公共训练循环

`train.py` 提供 `train_epoch(trainer, keys, values, order)`，按 order 逐对更新一轮（允许重复索引用于回放）。输入使用同设备 Tensor。函数不生成随机数、不创建 optimizer、不负责评估或保存；跨轮使用同一个 optimizer，保留其状态。Zero 基线可使用 ZeroTrainer 走同一训练循环。

```python
from associative_memory.train import DeltaTrainer, train_epoch

trainer = DeltaTrainer(model, lr=0.5)
train_epoch(trainer, keys, values, order)
```

NumPy 回归测试使用提交 `960150d` 的原始实现转为 FP32 作对照（`tests/reference/numpy_memory.py`），固定 seed=123，覆盖容量、遮挡和回放。检查每步 W、连续读出和符号差异。NumPy 与 PyTorch 即使同为 FP32 也不保证逐位相同；近零读出的符号可能因舍入变化，测试会报告这类差异，不要求正确率完全相等。

`Trainer` 组合 model、criterion 和 optimizer；`step(inputs, target)` 通过 `model(**inputs)` 计算，执行清梯度、反向传播和更新，返回脱离计算图的更新前损失。推理仍直接调用 model；快照分别使用 model 和 optimizer 的原生接口。

作业直接使用预配置的 Trainer，无需手动选择损失或创建 SGD：

```python
from associative_memory.train import DeltaTrainer, HebbianTrainer, train_epoch

trainer = DeltaTrainer(model, lr=0.5)  # 或 HebbianTrainer(model, lr=0.5)
trainer.step({"q": k}, v)
# 完整一轮：train_epoch(trainer, keys, values, order)
```

两者继承 Trainer，仅配置各自目标函数和无动量、无权重衰减的 SGD；共用 step 实现。模型与 optimizer 仍可通过 trainer.model 和 trainer.optimizer 访问。

每个规则的 loss 定义在对应 Trainer 子类中，不再提供独立 losses.py。公共 Trainer 仍支持传入其他 criterion 和 optimizer。

`ZeroTrainer(model)` 接受同样的 `step(inputs, target)` 和 `train_epoch` 调用，校验输入但不反向传播、不创建 optimizer、不修改 W，step 返回 None（没有训练目标）。它不会主动清零已有权重；零基线应传入新建的零初始化模型。
