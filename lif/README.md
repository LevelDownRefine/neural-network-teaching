# 单神经元 LIF 仿真

使用 snnTorch `Leaky` 实现 Euler 更新，以静息电位为零点；电压使用 mV、电流使用 nA、电阻使用 MΩ、时间使用 ms。达到阈值即发放并重置，随后在不应期内保持重置电位。

```bash
uv run --locked --extra cpu python lif/simulate.py
```

结果写入 `lif/output/`：恒定和周期电流四联图、阈值／电流／膜时间常数对照图，以及含脉冲数、平均发放率、首次脉冲时间的 `summary.json`。发放率以 50 ms 滑动窗口内实际脉冲数计算。

默认阈值 −50 mV、静息电位 −65 mV、电阻 10 MΩ，理论临界恒定电流为 1.5 nA；有限仿真中临界电流不发放。
