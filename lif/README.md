# Homework 2 · Section 1：LIF

[作业题目](homework-2.pdf)。环境：`uv sync --locked --extra cpu`。

| 项目 | 入口 |
| --- | --- |
| 模型与四信号展示 | [Notebook](neuron_dynamics.ipynb)、[Rust 模拟器](simulator/README.md) |
| 实验一：阈值 | [说明与结果](experiment1/README.md) |
| 实验二：电流 | [说明](experiment2/README.md) |
| 实验三：膜时间常数 | [说明](experiment3/README.md) |
| 扩展 A：突触电流 | [说明](bonus_a/README.md) |

模型保留在 model.py，绘图保留在 plot_lif.py；各实验的输出统一放 output 对应目录。Notebook 使用项目环境内核，从头执行。
