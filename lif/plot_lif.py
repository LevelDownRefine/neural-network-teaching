"""绘制单神经元输入、膜电位、脉冲及发放率。"""

import matplotlib.pyplot as plt
import numpy as np


def plot_signal(ax, signal, data, V_rest, V_th):
    time_ms = data["time_ms"]
    if signal == "current":
        ax.plot(data["input_time_ms"], data["current"])
        ax.set_ylabel("Current (nA)")
    elif signal == "voltage":
        ax.plot(time_ms, data["voltage"], label="Membrane potential")
        ax.axhline(V_th, color="red", linestyle="--", label="Threshold")
        ax.axhline(V_rest, color="green", linestyle=":", label="Resting potential")
        ax.set_ylabel("Voltage (mV)")
        ax.legend()
    elif signal == "spikes":
        ax.eventplot(time_ms[np.asarray(data["spikes"], dtype=bool)])
        ax.set(ylabel="Spike", yticks=[])
    elif signal == "rate":
        ax.plot(time_ms, data["rate"])
        ax.set_ylabel("Firing rate (Hz)")
    else:
        raise ValueError(f"Unknown signal: {signal}")
    ax.set_xlim(0, 200)


def plot_individual(data, signal, V_rest, V_th):
    fig, ax = plt.subplots(figsize=(10, 2))
    plot_signal(ax, signal, data, V_rest, V_th)
    ax.set_xlabel("Time (ms)")
    plt.show()


def plot_summary(name, data, V_rest, V_th):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, signal in zip(axes, ("current", "voltage", "spikes", "rate"), strict=True):
        plot_signal(ax, signal, data, V_rest, V_th)
    axes[0].set_title(name)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    plt.show()
