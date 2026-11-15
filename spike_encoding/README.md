# MNIST 脉冲编码实验

Task 1 见 [task1.ipynb](task1.ipynb)。

Task 2 使用相同 TSSL 结构、50,000／10,000 训练／验证划分、seed 42、增强与 EMA，分别训练 Rate、Latency 和 Phase，时间步数为 10、20、50、100。每组训练 100 轮，以验证集准确率选模后评估完整测试集。训练目标在第 0 步为 0，其余步为 1；损失按时间步数归一化。

Phase 的周期固定为 20 步；10 步实验只观察周期前半段。Population 需要多个输入通道，未纳入单通道结构的本组对照。

在项目根目录执行 `python spike_encoding/run_task2.py`，依次运行全部 12 组。结果保存在 `mnist/results/snntorch_task2_*`，已有完整结果会校验配置后跳过；中断后可再次运行以续训。每组记录测试分类指标、输入层与第一卷积层的实际 Spike 数，以及同设备同批量的平均批次推理时间，分别给出包含和不包含编码的结果；数据加载与设备传输不计入。汇总 CSV 和图表位于 `mnist/results/snntorch_task2_grid_seed42/`。
