# LIF 神经元仿真

[查看作业题目](homework-2.pdf)。

打开 [task1_lif.ipynb](task1_lif.ipynb)，使用项目 Python 3.12 环境依次运行。Notebook 从 Euler 更新开始，展示膜电位、阈值、重置和不应期；后续实验不包含在本文件中。

```bash
uv sync --locked --extra cpu
```

在 VS Code 等支持 Jupyter 的编辑器中选择项目的 `.venv` 内核。

## Rust 仿真

Notebook 和训练使用 snnTorch；可选的 Python 扩展复用 Rust 仿真核心。运行 uv sync --locked --extra cpu --extra rust 可通过 maturin 编译扩展，需要 Rust 工具链（Windows 还需 MSVC Build Tools）。原生界面不需要此扩展或 Python 环境。

测试分三层：cargo test 验证核心阈值、重置、不应期和非法输入；pytest 对照 snnTorch 的恒定、正弦及随机输入，逐元素检查膜电位和脉冲；原生界面测试检查布局、参数更新和导出。性能使用 release 构建独立测量，不用机器相关的耗时阈值判断 CI 是否通过。

本机 20,000 步实验，Rust Python 接口端到端约 1.2–1.3 ms；相同输入下原实现约 3.3–3.5 s。此处仅比较仿真，不包含绘图，也不代表其他硬件的耗时。

## Rust 原生界面

~~~bash
cargo run --release --locked --manifest-path lif/gui-rust/Cargo.toml
~~~

程序名为 LIF，采用中文深色界面。参数改变后立即更新四联图和统计，计算与界面均为 Rust，无需 Python 环境。鼠标悬停可查看图上数值；导出会在 lif/output 下建立独立目录，保存 trace.csv 和 parameters.json，不覆盖已有结果。

Windows 构建后可直接运行 lif/gui-rust/target/release/lif.exe。编译需要 Rust 1.88 或更新版本。
