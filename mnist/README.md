# MNIST 十分类实验

比较多类逻辑回归、连续激活 CNN 和 TSSL-BP 脉冲神经网络。实验保存测试准确率、混淆矩阵、各类别 precision／recall／F1 和 OvR AUC。

## 本机实验结果

| 模型 | 可训练参数 | 测试准确率 | Macro OvR AUC | 说明 |
|---|---:|---:|---:|---|
| 逻辑回归 | 7,850 | 92.41% | 0.993785 | 线性基线 |
| M5 CNN | 1,128,180 | 99.69% | 0.999961 | 数据增强和 EMA |
| 小 M5 CNN | 210,376 | 99.66% | 0.999994 | 参数量匹配 TSSL |
| TSSL 结构 CNN | 210,375 | 99.35% | 0.999971 | 连续激活，结构匹配 TSSL |
| TSSL-BP SNN | 210,375 | 99.27% | 0.998658 | 脉冲模型 |

每项为单次实验，最佳模型由验证集选出，测试集有 10,000 张图片。TSSL 结构 CNN 和 TSSL-BP 均使用 seed 42 的 50,000／10,000 训练／验证划分；其他模型的训练配置有差异。完整结果及权重保存在本机 `results/`，较新的大文件和实验输出未纳入 Git；仓库中 `results/comparison/` 是旧版结果，不包含上表所有实验。AUC 对模型类别得分应用 softmax 后计算。

另有 TSSL 官方协议实验：使用全部 60,000 张训练图并按测试集结果选模，得到 99.36%、AUC 0.999066。其测试集参与选模，不应与独立验证集实验视为同等条件。参数量为 1,128,175 的大 TSSL 在第 55 轮后暂停，保留本机检查点。

## 运行

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)，在仓库根目录执行：

```bash
uv sync --locked
uv run --locked python mnist/prepare_data.py
uv run --locked python mnist/train_logistic.py --epochs 20
uv run --locked python mnist/train_cnn.py --epochs 150 --augment --ema --tag repro_seed42
uv run --locked python mnist/train_cnn.py --architecture m5_small --epochs 150 --augment --ema --tag matched_seed42
uv run --locked python mnist/train_cnn.py --architecture tssl_cnn --epochs 100 --batch-size 50 --lr 0.0005 --seed 42 --tag aligned_seed42
uv run --locked python mnist/train_tssl.py --epochs 100 --seed 42 --tag repro_seed42
uv run --locked pytest -q
```

TSSL 需要 CUDA。大 TSSL 使用 `train_tssl.py --hidden-size 1712 --epochs 100 --seed 42 --tag matched_seed42`；相同参数加 `--resume` 可续训。训练脚本按轮保存检查点；`--max-batches` 只用于小批次代码检查，不能用于性能评估。

结构对齐 CNN 保留 TSSL 的两层卷积、固定权重求和池化、两层全连接、无偏置与相同参数量，并使用相同输入标准化、AdamW 和梯度及权重裁剪。CNN 使用 ReLU 和交叉熵，TSSL 使用脉冲神经元与脉冲核损失，二者仍存在训练机制差异。

TSSL 官方协议可用 `train_tssl.py --official-protocol --epochs 100 --seed 3 --tag official_seed3` 运行。该模式逐轮按测试集准确率选模型；适配版显式固定 PyTorch CPU 随机种子并使用零数据加载工作进程，官方脚本只显式设置 CUDA 和 NumPy 种子、使用四个工作进程。官方仓库当前配置文件为 200 轮，这里使用论文报告的 100 轮。核对源码版本为 `ffaac71d3344172980f048d98b123e40c5420401`。

`evaluate.py report` 汇总仓库中的旧结果，`evaluate.py plot` 绘图，`evaluate.py auc logistic_kair` 计算逻辑回归 AUC。`data_pipeline/` 管理划分和训练增强，`utils/classification_metrics.py` 提供公共指标。

## 来源

- CNN：An 等，[An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition](https://arxiv.org/abs/2008.10400)，[作者代码](https://github.com/ansh941/MnistSimpleCNN)。
- TSSL-BP：Zhang 与 Li，[Temporal Spike Sequence Learning via Backpropagation](https://papers.nips.cc/paper/2020/file/8bdb5058376143fa358981954e7626b8-Paper.pdf)，[作者代码](https://github.com/stonezwr/TSSL-BP)，MIT 许可。

逻辑回归是选用的线性基线，与经典硬阈值感知机并不完全相同。依赖锁定 PyTorch 2.11.0 和 torchvision 0.26.0：Windows 使用 CUDA 12.8 构建，Linux 使用 CPU 构建执行轻量测试。
