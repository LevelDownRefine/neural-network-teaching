# 实验一：单遍写入与完整线索检索

从仓库根目录运行以下命令。Hebbian、Delta 和零矩阵使用同一批随机键值，逐对写入一次，写完后用完整线索检索。键维数为 32、64、128，记忆数为 8、16、32、64、128，内容维数为 32，学习率为 0.5。每个组合独立重复 200 次。

```shell
uv run --locked --extra cpu python -m associative_memory.experiment1.run
```

结果位于 `associative_memory/output/experiment1/`。主图 `experiment1_curves.png` 按键维数分行，各线代表一种写入方法。左列画比特正确率随记忆负载 P/d 的变化，中列和右列分别画 MSE 和权重 Frobenius 范数随记忆数 P 的变化。表格 `experiment1_summary.csv` 保存每个条件的均值、样本标准差和权重字节数。

![实验一：比特正确率、MSE 和权重范数](../output/experiment1/experiment1_curves.png)

