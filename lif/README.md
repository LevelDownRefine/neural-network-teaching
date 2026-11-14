# LIF 神经元仿真

[查看作业题目](homework-2.pdf)。

打开 [task1_lif.ipynb](task1_lif.ipynb)，使用项目 Python 3.12 环境依次运行。Notebook 从 Euler 更新开始，展示膜电位、阈值、重置和不应期；后续实验不包含在本文件中。

```bash
uv sync --locked --extra cpu
```

在 VS Code 等支持 Jupyter 的编辑器中选择项目的 `.venv` 内核。

## 桌面交互

使用 uv 管理的独立 Python 3.12 环境安装并启动，避免已有 Conda 环境中的 Qt DLL 冲突：

~~~bash
uv sync --locked --managed-python --python 3.12 --extra cpu --extra gui
uv run --locked --extra cpu --extra gui python -m lif.gui
~~~

窗口中直接显示输入电流、膜电位、脉冲序列和发放率。可选择恒定或正弦输入，调整电流、阈值、时间常数、电阻、不应期和仿真时长；松开滑块后自动重新计算。计算在后台运行，连续修改参数时只展示最新结果。

静息电位固定为 −65 mV，时间步为 0.1 ms。统计覆盖完整仿真时长，显示范围可选择前 200 ms、前 500 ms 或全部。发放率使用约 100 ms 的回看窗口；起始阶段按已有时长计算。

可选的“导出结果”保存逐时间步 CSV、参数 JSON 和当前显示范围的 PNG；每行电流取步首值，膜电位和脉冲取步末值。仿真与 Notebook 共用 model.py，绘图共用 plot_lif.py。

## Rust 仿真

GUI 使用 Rust 前向循环；Notebook 和训练仍可使用 snnTorch。安装 GUI 依赖时会通过 maturin 编译本地扩展，需要 Rust 稳定工具链（Windows 还需 MSVC Build Tools）。单独安装扩展可运行 uv sync --locked --extra cpu --extra rust。

测试分三层：cargo test 验证核心阈值、重置、不应期和非法输入；pytest 对照 snnTorch 的恒定、正弦及随机输入，逐元素检查膜电位和脉冲；GUI 测试检查 Python 接口及异步显示。性能使用 release 构建独立测量，不用机器相关的耗时阈值判断 CI 是否通过。

本机 20,000 步实验，Rust Python 接口端到端约 1.2–1.3 ms；相同输入下原实现约 3.3–3.5 s。此处仅比较仿真，不包含绘图，也不代表其他硬件的耗时。
