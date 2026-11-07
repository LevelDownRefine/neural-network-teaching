# MNIST 十分类实验

比较逻辑回归、CNN 和 TSSL-BP 脉冲网络，评估测试准确率、混淆矩阵、各类别 precision／recall／F1 和 OvR AUC。

## 实验结果

| 模型 | 可训练参数 | 测试准确率 | Macro OvR AUC | 理论浮点运算次数（28×28） |
|---|---:|---:|---:|---:|
| 逻辑回归 | 7,850 | 92.41% | 0.993785 | 15,680 |
| M5 CNN（增强 + EMA） | 1,128,180 | 99.69% | 0.999961 | 274,739,200 |
| TSSL 结构 CNN（增强 + EMA） | 210,375 | 99.58% | 0.999979 | 2,742,000 |
| TSSL-BP SNN（增强 + EMA） | 210,375 | 99.38% | 0.998782 | 13,710,000 |

以上均为单次实验，以验证集选模、在 10,000 张官方测试图上评估。结构对齐的 CNN 和 TSSL 使用 seed 42、50,000／10,000 的训练／验证划分；其他训练配置有所不同。AUC 使用类别得分的 softmax 值。

TSSL 结构 CNN 与 SNN 的卷积、池化、全连接尺寸和参数量一致；CNN 使用 ReLU 和交叉熵，SNN 使用脉冲神经元和脉冲核损失。

运算次数只统计单个输入前向推理中可训练卷积和全连接层的理论密集浮点计算，按每次乘加约 2 次运算估算。TSSL 使用 5 个时间步；其输出经突触滤波，不能假定所有输入都是二值脉冲。统计不含偏置、归一化、池化、激活、softmax、突触滤波和神经元状态更新，也不含反向传播；因此不是完整推理运算量、实际耗时或能耗。

### 输入尺寸与运算量

令输入为单通道 n×n 图像，假设卷积结构和输出类别数不变，并调整首个全连接层以匹配展平后的特征，则上面统计范围内的运算次数近似为：

| 模型 | 浮点运算次数 |
|---|---|
| 逻辑回归 | 20n² |
| M5 CNN | 1,600(n−4)² + 102,400(n−8)² + 307,200(n−12)² + 614,400(n−16)² + 1,027,200(n−20)² |
| TSSL 结构 CNN | F(n) |
| TSSL-BP SNN | 5F(n) |

其中 `a = ⌊(n−4)/2⌋−4`，`b = ⌊a/2⌋`，`F(n) = 750(n−4)² + 30,000a² + 24,000b² + 6,000`。公式适用于上述结构的有效输入尺寸（四种模型一起比较时 n≥21），代入 n=28 得到上表。当前全连接层按 28×28 配置；尺寸变化若改变展平特征长度，须调整全连接层，参数量也会改变且原权重不再兼容。

TSSL 官方协议实验使用全部 60,000 张训练图、按测试集准确率选模，结果为 99.36%、AUC 0.999066；测试集参与选模，不宜与上表直接比较。

[混淆矩阵和逐类指标](RESULTS.md)列出了上表各模型的完整结果。较新的权重和原始结果保存在本机 `results/`，不纳入 Git。

## 安装与训练

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。以下命令使用 PyTorch 2.15 和 CUDA 13.2；仅使用 CPU 时将 `--extra cuda` 替换为 `--extra cpu`。TSSL 训练需要 CUDA。

```bash
uv sync --locked --extra cuda
uv run --locked --extra cuda python mnist/prepare_data.py
uv run --locked --extra cuda python mnist/train_logistic.py --epochs 20
uv run --locked --extra cuda python mnist/train_cnn.py --epochs 150 --augment --ema --tag repro_seed42
uv run --locked --extra cuda python mnist/train_cnn.py --architecture tssl_cnn --epochs 100 --batch-size 50 --lr 0.0005 --seed 42 --augment --ema --tag augment_ema_seed42
uv run --locked --extra cuda python mnist/train_tssl.py --epochs 100 --seed 42 --augment --ema --tag augment_ema_seed42
uv run --locked --extra cuda ruff check .
uv run --locked --extra cuda ruff format --check .
```

CNN 和 TSSL 按轮保存检查点；用相同参数加 `--resume` 续训。`--max-batches` 仅用于代码检查。

TSSL 可加 `--ema` 使用权重平均，加 `--augment` 启用训练集旋转和平移；这两个选项不能用于 `--official-protocol`。

官方协议使用 `train_tssl.py --official-protocol --epochs 100 --seed 3 --tag official_seed3`。本实现固定 PyTorch CPU 随机种子、使用 0 个数据加载进程；官方脚本只显式设置 CUDA 和 NumPy 种子、使用 4 个进程。官方配置文件为 200 轮，此处采用论文的 100 轮；核对的源码版本为 `ffaac71d3344172980f048d98b123e40c5420401`。

## 评估

- `uv run --locked --extra cuda python mnist/evaluate.py report`：汇总已完成实验，记录 seed 和选模方式。
- `uv run --locked --extra cuda python mnist/evaluate.py plot`：绘制混淆矩阵。
- `uv run --locked --extra cuda python mnist/evaluate.py complexity`：计算现有 28×28 模型各层理论浮点运算次数。
- `uv run --locked --extra cuda python mnist/evaluate.py auc logistic_kair`：计算旧逻辑回归权重的 AUC。

评估已有权重，例如：

```bash
uv run --locked --extra cuda python mnist/evaluate.py checkpoint --model tssl_cnn --weights mnist/results/cnn_tssl_cnn_augment_ema_seed42/best.pt --output evaluation.json
```

`--model` 可选 `logistic`、`m5`、`tssl_cnn`、`tssl`。单独评估不重新选模；数据划分和选模方式以原实验记录为准。历史记录中无法确定的字段标为未知，smoke 结果不进入汇总。

`utils/` 是内部模块，评估请通过 `mnist/evaluate.py` 调用。

## 参考

- M5 CNN：[论文](https://arxiv.org/abs/2008.10400)、[作者代码](https://github.com/ansh941/MnistSimpleCNN)。
- TSSL-BP：[论文](https://papers.nips.cc/paper/2020/file/8bdb5058376143fa358981954e7626b8-Paper.pdf)、[作者代码](https://github.com/stonezwr/TSSL-BP)（MIT）。

## Offline 推理基准

WSL2 CUDA 环境（需支持 CUDA 的驱动和 C 编译器）：

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --locked --extra cuda
uv run --locked --extra cuda python mnist/benchmark.py offline --model tssl --weights mnist/results/tssl_augment_ema_seed42/best.pt --batch-size 512 --engine inductor --seconds 30 --output mnist/results/tssl_augment_ema_seed42/offline_inductor.json
```

提前保存编译产物，之后加载并在连接好测试电源后按回车开始测量：

```bash
uv run --locked --extra cuda python mnist/benchmark.py power --model tssl --weights mnist/results/tssl_augment_ema_seed42/best.pt --batch-size 512 --engine inductor --save-compiled mnist/results/tssl_aot
uv run --locked --extra cuda python mnist/benchmark.py power --model tssl --weights mnist/results/tssl_augment_ema_seed42/best.pt --batch-size 512 --engine inductor --load-compiled mnist/results/tssl_aot --seconds 60
```

产物依赖模型代码和环境；更换代码或环境后应重新生成。保存与加载时模型、权重、batch、PyTorch 和设备需一致。等待期间不采集功率或计时；外接电表按回车时同步开始记录。结果中 `throughput.seconds` 为回车后推理测量的实际用时，`throughput.images` 为处理图片数；实际用时可能略长于 `--seconds`，因为会等待当前 batch 完成。`power` 也可直接使用 eager 或普通 Inductor，无须保存产物。

WSL 使用独立的 `.venv-wsl`，与 Windows 的 `.venv` 分开。`--engine inductor` 使用 `torch.compile(fullgraph=True)`，编译和预热时间单独记录，不计入吞吐量；默认 `eager`。精度使用 FP32 和后端默认设置。

图像预先放在设备上；预热后连续推理，报告吞吐量、PyTorch 峰值分配显存和 GPU 板卡平均功率。计时包含模型调用、GPU 同步和 TSSL 时间步展开，不含数据加载、预处理和传输。`nvidia-smi` 默认约每秒采集功率；平均功率乘运行时间估算板卡能耗，包含 GPU 空闲和其他任务功耗，不含 CPU 等整机用电。少于两次有效采样时不估算功率和能耗。

以下为 RTX 5070 Ti Laptop GPU 的单次 FP32 实验，使用默认 NCHW 输入；每档吞吐量测约 30 秒，编译和预热时间不计入。逻辑回归使用 Windows、PyTorch 2.11.0+cu128 和 eager；其余模型使用 WSL2、PyTorch 2.15.0+cu132 和 Inductor。运行环境不同，结果需谨慎比较。batch 沿用此前 FP32 扫描中吞吐量最高的档位，不表示 Inductor 的最优 batch。

| 模型 | 执行方式 | batch | 吞吐量（图片/秒） | 平均板卡功率（W） | 板卡能耗估算（mJ/图） |
|---|---|---:|---:|---:|---:|
| 逻辑回归 | eager | 2048 | 22,289,579 | 63.8 | 0.00286 |
| M5 CNN | Inductor | 128 | 51,530 | 108.6 | 2.107 |
| TSSL 结构 CNN | Inductor | 1024 | 1,328,729 | 104.6 | 0.0787 |
| TSSL-BP SNN | Inductor | 512 | 84,937 | 110.3 | 1.298 |


### 整机耗电测试

两次实验分别于 2026-11-06 和 2026-11-07 进行，使用上述增强 + EMA 权重，在 RTX 5070 Ti Laptop GPU、WSL2、PyTorch 2.15.0+cu132 上以 FP32 Inductor 运行。编译产物加载、预热后等待接入电表；计时 15 分钟，重复推理已放在 GPU 上的同一批 MNIST 测试图片，不含数据加载和传输。电表测量整机耗电，数值由使用者记录，仪器标称精度为 0.02 度。

| 模型 | batch | 实际用时（秒） | 累计图片数 | 吞吐量（图片/秒） | 电表耗电（度） | 每百万张耗电（度） | GPU 平均板卡功率（W） |
|---|---:|---:|---:|---:|---:|---:|---:|
| TSSL 结构 CNN | 1024 | 900.000 | 1,218,428,928 | 1,353,810 | 0.05 | 0.0000410 | 124.2 |
| TSSL-BP SNN | 512 | 900.001 | 67,577,856 | 75,086 | 0.05 | 0.000740 | 121.9 |

单位图片整机耗电比（SNN ∶ CNN）约为 18 ∶ 1。

单位图片耗电等于整机电表耗电除以累计推理图片数，包含空闲及其他任务的耗电。电表精度相对单次读数较粗，结果为近似值；不同 batch、测试日期和运行状态也会影响比较。电表读数与 GPU 板卡功率为不同测量范围。完整采样和原始计数见 [CNN](results/power/tssl_cnn_15min.json) 和 [SNN](results/power/tssl_snn_15min.json)。
