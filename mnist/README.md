# MNIST 十分类实验

以多类逻辑回归、连续激活 CNN 和脉冲神经网络对照分类表现，并保留模型实现与训练协议差异。

## 已有结果

| 模型 | 测试准确率 | OvR Macro-AUC | 状态 |
|---|---:|---:|---|
| 多类逻辑回归 | 92.41% | 0.993785 | 已完成，单次运行 |
| 单个 M5 CNN | 99.52% | 0.999946 | 已完成，单次运行 |
| TSSL-BP SNN | 99.32% | 0.999115 | 中断训练后选择第 53 轮权重评估 |
| Spike-based BP SNN | 待完成 | 待完成 | 正式训练中 |

各类别指标、混淆矩阵和准确率在 `results/comparison/`；完整结果与最佳权重位于各模型结果目录。AUC 使用类别得分经 softmax 后计算。TSSL 使用 seed 3 划分训练与验证集，逻辑回归和 CNN 使用 seed 42，不能视为完全受控的代际比较。

CNN 使用论文的单个 M5 架构，但未使用论文集成、数据增强或 EMA，不能将结果作为论文 99.91% 集成准确率的复现。TSSL 的优化器和随机数状态未保存，无法精确续训；SpikeBP 的新训练脚本按轮保存完整检查点。

## 安装与使用

Python 3.12；使用 [uv](https://docs.astral.sh/uv/) 管理依赖与锁文件。在仓库根目录执行：

```bash
uv sync --locked
uv run --locked python mnist/prepare_data.py
uv run --locked python mnist/train_logistic.py --epochs 20
uv run --locked python mnist/train_cnn.py --epochs 10
uv run --locked python mnist/train_tssl.py --epochs 100
uv run --locked python mnist/train_spikebp.py --epochs 125
uv run --locked python mnist/evaluate.py report
uv run --locked python mnist/evaluate.py plot
uv run --locked pytest -q
```

后两种 SNN 需要 CUDA；SpikeBP 可用相同参数加 `--resume` 从最近完成的一轮恢复。数据、缓存及训练中的检查点不进入 Git，`prepare_data.py` 只下载 MNIST。`models/` 包含 M5 结构的独立实现和实验需要的 SNN 代码；SpikeBP 源码已直接适配新版 Python/PyTorch。SNN 作者代码保留各自 MIT 许可证。

依赖锁定 PyTorch 2.11.0 和 torchvision 0.26.0：Windows 选择 CUDA 12.8 构建（需要兼容的 NVIDIA 驱动），Linux 选择 CPU 构建，用于轻量测试；Linux 上运行 SNN 需要另行配置匹配的 CUDA PyTorch 环境。GitHub Actions 在 CPU 上运行测试，不下载 MNIST 或执行完整训练。

## 来源

- CNN：An 等，[An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition](https://arxiv.org/abs/2008.10400)，[作者代码](https://github.com/ansh941/MnistSimpleCNN)。
- TSSL-BP：Zhang 与 Li，[Temporal Spike Sequence Learning via Backpropagation](https://papers.nips.cc/paper/2020/file/8bdb5058376143fa358981954e7626b8-Paper.pdf)，[作者代码](https://github.com/stonezwr/TSSL-BP)，MIT 许可。
- Spike-based BP：Lee 等，[Enabling Spike-Based Backpropagation](https://doi.org/10.3389/fnins.2020.00119)，[作者代码](https://github.com/chan8972/Enabling_Spikebased_Backpropagation)，MIT 许可。

逻辑回归是选用的线性基线，与经典硬阈值感知机并不完全相同。`utils/common_data.py` 与 `utils/classification_metrics.py` 提供公共数据划分及评估，模型训练算法分别保留。`evaluate.py report` 汇总已有结果，`evaluate.py auc logistic_kair` 计算逻辑回归 AUC，`evaluate.py tssl` 评估已保存的 TSSL 最佳权重。
