# Homework 2：结果整理方案

以 homework-2.pdf 的题号为目录，报告正文只保留设置、关键图表、结论；完整参数、轨迹、训练日志通过链接提供。这里是现有材料盘点，不代表尚未核实的实验已经完成。

## 建议的最终结构

1. Section 1：LIF 实现与三个对照实验。
2. Section 1 Bonus：只选实际完成且证据齐全的一项。
3. Section 2 Task 1：Rate、Latency、Phase 原理与同图 Raster。
4. Section 2 Task 2：准确率、输入/隐藏层脉冲数量、同设备推理时间。
5. Section 2 Task 3：SHD 数据特点、小规模实验与局限。
6. 附录：运行命令、软硬件环境、原始数据索引。

## 现有证据与整理位置

| 作业要求 | 已有材料 | 报告中怎么呈现 |
| --- | --- | --- |
| LIF Euler 更新、阈值、重置、不应期 | [基础 Notebook](neuron_dynamics.ipynb)、[模型](model.py) | 一组方程和一幅膜电位/脉冲图 |
| 恒定/周期输入、四信号可视化 | [模拟器](simulator/README.md) | 固定参数截图或导出图；交互程序作为代码附件 |
| 阈值实验 | [结果分析](threshold_effect/results/analysis.md)、[CSV](threshold_effect/results/summary.csv) | 一张对照图、一张精简表、三条题目答案 |
| 电流实验 | 需核实固定参数运行、保存图表与统计数据的位置 | 独立列出临界输入、ISI、平均发放率，不能只用交互演示代替 |
| 膜时间常数实验 | [编码 Notebook 最后一节：LIF 对照实验](../spike_encoding/encoding_and_membrane_dynamics.ipynb) | 同一输入下曲线、首发时间、总脉冲数与平均发放率；解释 tau=RC |
| Bonus | 需确认选项和实际结果 | 只保留完成的一项及对应证据 |
| 三种编码原理与可视化 | [Task 1 Notebook](../spike_encoding/encoding_and_membrane_dynamics.ipynb) | 三行原理/公式表、同图同窗口 Raster、脉冲数 |
| 编码分类对照 | [Task 2 完整结果](../spike_encoding/encoding_classification_results.md) | 正文保留 3×4 准确率表与两张关键图，其余表移附录 |
| SHD 探索 | [实验说明](../spike_encoding/shd_baseline/README.md)、[结果分析](../spike_encoding/shd_baseline/results/analysis.md) | 原始事件 Raster、准确率曲线、测试结果与边界 |

## 已核实的主要结果

- 阈值：固定输入 2 nA，阈值 -55/-50/-46 mV 时平均发放率为 62.5/33.5/16 Hz；-45/-44 mV 在 2 s 内无脉冲。完整解释引用原分析，区分平衡电位和有限观测窗口。
- MNIST：Rate/Latency/Phase 均完成 10/20/50/100 步配置。最高测试准确率为 Rate 50 步的 99.57%；Latency 100 步为 95.57%。这是单种子观察，不写“统计显著”。
- 推理时间：同一 V100、batch=20，纯模型每批约从 10 步的 0.76–0.78 ms 增至 100 步的 10.93–10.99 ms。明确不含编码与传输，脉冲数不等同能耗。
- SHD：3000/500/500 训练/验证/测试子集，700→128 LIF→20，测试 48.2%（241/500）；不是完整官方测试集成绩。

## 正文精简原则

每个实验统一为“设置 → 图/表 → 回答题目”，避免逐次记录操作过程。环境只集中写一次，特殊设备差异在相应指标旁注明。训练轮数、早停与手动停止按实际记录写，不能把策略相同误写成训练轮数完全一致。

已有 Task 2 长文作为详细结果页，正文引用而不复制全部内容。MNIST 的其他基线、功耗测量、关联记忆、HH 仿真不混入 Homework 2；额外论文复现如需展示，放附录并标注与基础 SHD 实验的区别。

最终提交需要 PDF 报告与代码 ZIP。先补齐电流、Bonus 的结果定位，再生成 PDF，避免报告存在只有结论而没有运行证据的部分。

## 代码与文档约定

- 从仓库根目录运行命令；公共环境统一使用 `uv sync --locked --extra cpu --dev`，Notebook 选择同一环境内核。SHD 工具还需要 h5py，按具体实验说明安装。
- Python 和 Notebook 遵循根目录 pyproject.toml 的 Ruff 配置：函数/变量 snake_case、常量 UPPER_CASE、导入分组由 Ruff 管理。物理公式已有的 V、I、R、tau 等符号保留，避免无意义的接口重命名。
- 文档统一顺序：任务与入口 → 运行方法 → 输出位置 → 结果与局限。每项结果记录 seed、数据划分、实际轮数、单位和设备。
- 路径使用相对于文档的链接；生成数据保持原位置，避免搬动导致 Notebook 和脚本失效。
- 格式检查：`uv run --locked --extra cpu ruff check lif spike_encoding` 和 `uv run --locked --extra cpu ruff format --check lif spike_encoding`。

## 补充入口

| 材料 | 位置 | 用途 |
| --- | --- | --- |
| Section 1 Experiment 3 | [encoding_and_membrane_dynamics.ipynb 最后一节](../spike_encoding/encoding_and_membrane_dynamics.ipynb) | 固定 2 nA、-50 mV，对比 tau=10/20/40 ms |
| Section 2 Task 3 论文复现 | [shd_spikcommander_evaluation.ipynb](../spike_encoding/shd_spikcommander_evaluation.ipynb)、[评估脚本](../spike_encoding/evaluate_spikcommander.py) | 作者权重评估，10 ms Top-1 96.41%；与自训练小型 SHD 实验分开呈现 |
| 论文复现数据准备 | [prepare_shd.py](../spike_encoding/prepare_shd.py) | 下载并校验官方 SHD 文件，默认写入仓库父目录 outputs/Section2_Task3_SHD |

跨目录的实验暂以索引明确位置：膜时间常数虽然属于 Section 1，目前实际位于 spike_encoding/encoding_and_membrane_dynamics.ipynb，不应遗漏或误认为未完成。

## 复用与设计检查记录

- SHD 两个入口共用 prepare_shd.py 的校验下载和原子解压，数据目录保持兼容。下载时打印当前文件，完成后打印解压路径。
- LIF 参数扫描共用 model.py 的 simulate_experiment；阈值实验使用 summarize_spikes 统一统计全程发放率、首发和 ISI。plot_lif 支持 time_limit_ms=None 显示全程。
- SHD baseline 的固定窗口二值分箱与作者模型的事件锚定计数分箱不同，不合并预处理语义。
- Python 与 Rust 后端需要保留独立实现，并通过参考测试交叉验证，不能为了去重删除验证依据。
- 尚存结构问题：编码 Notebook 混有膜时间常数实验；基础 LIF 测试仍从 Notebook 提取代码；SHD 基础训练的配置与绘图集中在 main 中。后续拆分应单独进行，保证 Notebook 输出与结果记录一起更新。
