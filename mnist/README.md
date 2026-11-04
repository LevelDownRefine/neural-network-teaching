# MNIST 十分类实验

比较逻辑回归、CNN 和 TSSL-BP 脉冲网络，评估测试准确率、混淆矩阵、各类别 precision／recall／F1 和 OvR AUC。

## 实验结果

| 模型 | 可训练参数 | 测试准确率 | Macro OvR AUC |
|---|---:|---:|---:|
| 逻辑回归 | 7,850 | 92.41% | 0.993785 |
| M5 CNN（增强 + EMA） | 1,128,180 | 99.69% | 0.999961 |
| TSSL 结构 CNN | 210,375 | 99.35% | 0.999971 |
| TSSL-BP SNN | 210,375 | 99.27% | 0.998658 |

以上均为单次实验，以验证集选模、在 10,000 张官方测试图上评估。结构对齐的 CNN 和 TSSL 使用 seed 42、50,000／10,000 的训练／验证划分；其他训练配置有所不同。AUC 使用类别得分的 softmax 值。

TSSL 结构 CNN 与 SNN 的卷积、池化、全连接尺寸和参数量一致；CNN 使用 ReLU 和交叉熵，SNN 使用脉冲神经元和脉冲核损失。

TSSL 官方协议实验使用全部 60,000 张训练图、按测试集准确率选模，结果为 99.36%、AUC 0.999066；测试集参与选模，不宜与上表直接比较。

[混淆矩阵和逐类指标](RESULTS.md)列出了上表各模型的完整结果。较新的权重和原始结果保存在本机 `results/`，不纳入 Git。

## 安装与训练

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)；Windows 使用 CUDA 12.8 PyTorch，Linux 使用 CPU 构建。TSSL 训练需要 CUDA。

```bash
uv sync --locked
uv run --locked python mnist/prepare_data.py
uv run --locked python mnist/train_logistic.py --epochs 20
uv run --locked python mnist/train_cnn.py --epochs 150 --augment --ema --tag repro_seed42
uv run --locked python mnist/train_cnn.py --architecture tssl_cnn --epochs 100 --batch-size 50 --lr 0.0005 --seed 42 --tag aligned_seed42
uv run --locked python mnist/train_tssl.py --epochs 100 --seed 42 --tag repro_seed42
uv run --locked ruff check .
uv run --locked ruff format --check .
```

CNN 和 TSSL 按轮保存检查点；用相同参数加 `--resume` 续训。`--max-batches` 仅用于代码检查。

官方协议使用 `train_tssl.py --official-protocol --epochs 100 --seed 3 --tag official_seed3`。本实现固定 PyTorch CPU 随机种子、使用 0 个数据加载进程；官方脚本只显式设置 CUDA 和 NumPy 种子、使用 4 个进程。官方配置文件为 200 轮，此处采用论文的 100 轮；核对的源码版本为 `ffaac71d3344172980f048d98b123e40c5420401`。

## 评估

- `uv run --locked python mnist/evaluate.py report`：汇总已完成实验，记录 seed 和选模方式。
- `uv run --locked python mnist/evaluate.py plot`：绘制混淆矩阵。
- `uv run --locked python mnist/evaluate.py auc logistic_kair`：计算旧逻辑回归权重的 AUC。

评估已有权重，例如：

```bash
uv run --locked python mnist/evaluate.py checkpoint --model tssl_cnn --weights mnist/results/cnn_tssl_cnn_aligned_seed42/best.pt --output evaluation.json
```

`--model` 可选 `logistic`、`m5`、`tssl_cnn`、`tssl`。单独评估不重新选模；数据划分和选模方式以原实验记录为准。历史记录中无法确定的字段标为未知，smoke 结果不进入汇总。

`utils/` 是内部模块，评估请通过 `mnist/evaluate.py` 调用。

## 参考

- M5 CNN：[论文](https://arxiv.org/abs/2008.10400)、[作者代码](https://github.com/ansh941/MnistSimpleCNN)。
- TSSL-BP：[论文](https://papers.nips.cc/paper/2020/file/8bdb5058376143fa358981954e7626b8-Paper.pdf)、[作者代码](https://github.com/stonezwr/TSSL-BP)（MIT）。
