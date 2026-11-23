# Homework 3 方向 C：记忆串扰

从仓库根目录，在含 NumPy、PyTorch、Matplotlib 的环境中运行：

```powershell
python -m associative_memory.exploration_c.run
```

结论和推导见 `report.md`。图片单独保存在 `results/`，包括负载与维数、学习率、相关线索三张 PNG。`individual_runs.csv` 为逐种子结果，`summary.csv` 为 80 个种子的均值及样本标准差。`config.json` 记录参数和版本。

主实验固定学习率 0.5。学习率扫描仅验证理论缩放，不基于结果选择学习率。训练使用单遍 Hebbian 闭式解，并通过当前公共 PyTorch 模型与 Trainer 的实际训练结果交叉验证。
