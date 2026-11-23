# 实验二：线索不完整还能想起来吗

阅读 `report.md` 查看实验设置、结果、三组样例和解释。图片单独保存在 `../output/experiment2/*.png`。

## 复现

在仓库根目录使用包含 NumPy、Matplotlib、PyTorch 的 Python 环境执行：

```powershell
python -m associative_memory.experiment2.run
```

上述命令运行种子 42 的主实验和种子 0–19 的重复实验，输出数据和图像。
配置固定为作业要求，不进行测试集调参。

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 实验、统计、绘图与运行时检查 |
| `../output/experiment2/config.json` | 参数、种子、版本、原始数组的轴定义 |
| `../output/experiment2/main_metrics.csv` | 主实验两种方法、四种遮挡比例的指标 |
| `../output/experiment2/repeat_metrics.csv` | 20 组重复实验逐组指标 |
| `../output/experiment2/repeat_summary.csv` | 重复实验均值和样本标准差 |
| `../output/experiment2/main_raw.npz` | 配对、顺序、掩码、查询、权重及全部连续输出 |
| `../output/experiment2/main_curves.png` | 主实验正确率、MSE 曲线 |
| `../output/experiment2/repeat_curves.png` | 重复实验曲线，误差条为跨种子标准差 |
| `../output/experiment2/example_0.png` 至 `example_2.png` | 提前固定的三条记忆，各展示四种遮挡比例 |

`main_raw.npz` 中 `weights` 和 `z` 的方法轴依次为 Hebbian、Delta。
遮挡轴依次为 0、0.25、0.5、0.75，具体轴定义记录在 `config.json`。
相同种子下共享配对、写入顺序和掩码。0% 遮挡的十次查询完全相同。

已执行的检查包括掩码坐标数、查询单位范数、检索权重不变、Hebbian 闭式解。
完整恢复率是补充指标，指一次检索中 32 个内容比特全部正确。

