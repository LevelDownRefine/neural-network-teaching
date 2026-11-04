# MNIST 十分类实验

以多类逻辑回归、连续激活 CNN 和脉冲神经网络对照分类表现，并保留模型实现与训练协议差异。

## 已有结果

| 模型 | 测试准确率 | OvR Macro-AUC | 状态 |
|---|---:|---:|---|
| 多类逻辑回归 | 92.41% | 0.993785 | 已完成，单次运行 |
| 单个 M5 CNN | 99.52% | 0.999946 | 已完成，单次运行 |
| TSSL-BP SNN | 99.32% | 0.999115 | 中断训练后选择第 53 轮权重评估 |
| Spike-based BP SNN | 待完成 | 待完成 | 已停止训练 |

另一个 M5 单模型实验使用数据增强与 EMA，在 seed 42 的训练／验证划分下取得测试准确率 99.69%（单次运行）；新的 TSSL 同划分实验正在训练。

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

参考作者训练配置的单模型实验可运行：

```bash
uv run --locked python mnist/train_cnn.py --epochs 150 --augment --ema --tag repro_seed42
uv run --locked python mnist/train_tssl.py --epochs 100 --seed 42 --tag repro_seed42
uv run --locked python mnist/train_cnn.py --architecture sopcnn --epochs 10 --batch-size 64 --augment --tag pilot_seed42
```

CNN 的旋转和位移只作用于训练集；验证集用于选模，官方测试集仅在训练结束后评估。实现使用 torchvision 的随机增强，随机数机制与作者代码不同；保留 50,000/10,000 的训练／验证划分，因而这是对作者设置的近似复现，不是完全相同的实验。两条命令使用 `--resume` 并保留相同参数，可从最近完成的 epoch 恢复；各自输出目录与已有结果分开。若只跑少量批次检查代码，可用 `--max-batches` 和单独的 `--tag`，但这些结果不是性能指标。

SOPCNN 试验采用下述第三方实现的四层卷积、2048 维全连接层和 0.8 dropout，复用本项目的 Adam、数据增强、验证集选模及评估流程。第三方报告的 99.78% 使用 SGD + Lookahead；该试验没有复现其优化器或训练时长，也不等同论文的精确复现。10 轮仅用于初步检查，不能保证达到文献指标。

参数量对照增加两个独立模型：`--architecture m5_small` 将 M5 通道改为 14、30、37、48、69，共 210,376 个可训练参数，对照原 TSSL 的 210,375 个；TSSL 使用 `--hidden-size 1712` 将隐藏层从 300 扩大到 1712，共 1,128,175 个参数，对照 M5 的 1,128,180 个。两者各自沿用原模型训练设置，使用 seed 42 的同一数据划分和公共测试指标。参数量接近不意味着层结构、计算量或训练策略相同。

```bash
uv run --locked python mnist/train_cnn.py --architecture m5_small --epochs 150 --augment --ema --seed 42 --tag matched_seed42
uv run --locked python mnist/train_tssl.py --hidden-size 1712 --epochs 100 --seed 42 --tag matched_seed42
```

后两种 SNN 需要 CUDA；SpikeBP 可用相同参数加 `--resume` 从最近完成的一轮恢复。数据、缓存及训练中的检查点不进入 Git，`prepare_data.py` 只下载 MNIST。`models/` 包含 M5 结构的独立实现和实验需要的 SNN 代码；SpikeBP 源码已直接适配新版 Python/PyTorch。SNN 作者代码保留各自 MIT 许可证。

依赖锁定 PyTorch 2.11.0 和 torchvision 0.26.0：Windows 选择 CUDA 12.8 构建（需要兼容的 NVIDIA 驱动），Linux 选择 CPU 构建，用于轻量测试；Linux 上运行 SNN 需要另行配置匹配的 CUDA PyTorch 环境。GitHub Actions 在 CPU 上运行测试，不下载 MNIST 或执行完整训练。

为核对 TSSL 官方结果，可以单独运行 `uv run --locked python mnist/train_tssl.py --official-protocol --epochs 100 --seed 3 --tag official_seed3`。该模式使用全部 60,000 个官方训练样本，并参照官方训练脚本每轮在官方测试集选取最佳权重（相同准确率取较后一轮）。因此输出的测试准确率参与了模型选择，不能作为独立测试估计，也不能与使用独立验证集选模的实验直接当成同等条件比较。此适配显式固定 PyTorch CPU 随机种子并使用零数据加载工作进程；官方脚本只显式设置 CUDA 和 NumPy 种子、使用四个工作进程。官方仓库 YAML 当前默认 200 轮，此处使用论文报告的 100 轮。官方源码核对版本为 ffaac71d3344172980f048d98b123e40c5420401。

## 来源

- CNN：An 等，[An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition](https://arxiv.org/abs/2008.10400)，[作者代码](https://github.com/ansh941/MnistSimpleCNN)。
- SOPCNN：Assiri，[Stochastic Optimization of Plain Convolutional Neural Networks with Simple methods](https://arxiv.org/abs/2001.08856)，[第三方 PyTorch 实现](https://github.com/junaidaliop/MNIST-SOPCNN)。
- TSSL-BP：Zhang 与 Li，[Temporal Spike Sequence Learning via Backpropagation](https://papers.nips.cc/paper/2020/file/8bdb5058376143fa358981954e7626b8-Paper.pdf)，[作者代码](https://github.com/stonezwr/TSSL-BP)，MIT 许可。
- Spike-based BP：Lee 等，[Enabling Spike-Based Backpropagation](https://doi.org/10.3389/fnins.2020.00119)，[作者代码](https://github.com/chan8972/Enabling_Spikebased_Backpropagation)，MIT 许可。

逻辑回归是选用的线性基线，与经典硬阈值感知机并不完全相同。`data_pipeline/` 管理数据划分及训练增强；`utils/classification_metrics.py` 提供公共评估，模型训练算法分别保留。`evaluate.py report` 汇总已有结果，`evaluate.py auc logistic_kair` 计算逻辑回归 AUC，`evaluate.py tssl` 评估已保存的 TSSL 最佳权重。
