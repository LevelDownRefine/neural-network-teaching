# SHD 探索性实验

SHD 是公开的脉冲语音数据集；原始语音经人工耳蜗转换为 700 通道脉冲，任务是识别英语与德语数字的 20 个类别。官方训练集 8156 条、测试集 2264 条。数据来源及引用：[Zenke Lab](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)。

在项目根目录运行（需要 torch、snntorch、numpy、matplotlib 和 h5py）：

```powershell
conda activate hw
python spike_encoding/shd_baseline/download.py
python spike_encoding/shd_baseline/experiment.py
```

下载器将两个官方压缩包下载到 `shd_data/`，核对 MD5 后解压为 `shd_train.h5` 和 `shd_test.h5`。下载失败时可重新运行；也可以手动从官方数据源下载两个文件并解压到该目录。

默认从官方训练集随机抽取互不重叠的 3000 条训练、500 条验证，另从官方测试集抽取 500 条测试；固定随机种子 42。每 20 ms 将一个通道中的事件合并为 0/1，共 50 步，窗口外事件舍弃。模型为 700→128 LIF→20 全连接 SNN，利用替代梯度训练，按照验证集准确率选择模型，最后评估一次测试集。

输出在 `results/`：原始事件 Raster、准确率曲线、划分索引、最佳权重和包含配置与测试指标的 JSON。测试子集不是完整官方测试集，单次实验仅作为探索，不能和完整测试集结果直接比较。

参数可用 `--help` 查看；例如 `--epochs 20 --test-size 2264`。结果由程序生成，不随脚本附带预先计算的指标。
