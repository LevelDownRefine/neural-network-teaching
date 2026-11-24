import numpy as np

from .run import KEY_DIMS, REPEATS

METHODS = ("Delta", "PDN")
COLORS = {"Delta": "#B84828", "PDN": "#00766E"}
LABELS = {"Delta": "作业原版 Delta", "PDN": "论文方法 PDN（固定门值）"}


def setup_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A8B6BB",
            "axes.axisbelow": True,
            "font.size": 10,
        }
    )
    return plt


def plot_accuracy(rows, output):
    from matplotlib.ticker import PercentFormatter

    plt = setup_plots()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.1, 7.0),
        sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [1.8, 1]},
    )
    fig.subplots_adjust(
        left=0.092, right=0.978, bottom=0.165, top=0.755, hspace=0.25, wspace=0.21
    )
    fig.text(
        0.082,
        0.934,
        "原版 Delta 与固定门值 PDN 的单遍检索正确率",
        fontsize=19,
        weight="bold",
    )
    pdn_gains = [r["gain_vs_delta_pp"] for r in rows if r["method"] == "PDN"]
    fig.text(
        0.082,
        0.88,
        f"完整线索检索，每组 {REPEATS} 次配对重复。"
        f"{len(pdn_gains)} 组条件等权平均，PDN 较原版 Delta {np.mean(pdn_gains):+.2f} 个百分点。",
        color="#405560",
    )
    for col, d in enumerate(KEY_DIMS):
        top, diff = axes[:, col]
        top.set_title(f"键维度  d = {d}", loc="left", weight="bold", fontsize=11)
        for method, mark, line in zip(METHODS, ("o", "D"), ("--", "-"), strict=True):
            subset = [r for r in rows if r["d"] == d and r["method"] == method]
            load = [r["load"] for r in subset]
            top.plot(
                load,
                [r["bit_accuracy_mean"] for r in subset],
                line,
                color=COLORS[method],
                marker=mark,
                markersize=4.2,
                linewidth=1.65,
                label=LABELS[method],
            )
            if method == "PDN":
                diff.axhline(0, color="#76888D", linewidth=0.85, linestyle="--")
                diff.fill_between(
                    load,
                    [r["gain_vs_delta_ci95_low_pp"] for r in subset],
                    [r["gain_vs_delta_ci95_high_pp"] for r in subset],
                    color=COLORS[method],
                    alpha=0.20,
                )
                diff.plot(
                    load,
                    [r["gain_vs_delta_pp"] for r in subset],
                    color=COLORS[method],
                    marker=mark,
                    markersize=4.2,
                    linewidth=1.65,
                )
        for ax in (top, diff):
            ax.grid(axis="y", linewidth=0.6, color="#DEE5E8")
            ax.set_xscale("log", base=2)
        loads = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4)
        diff.set_xticks(loads, ("1/16", "1/8", "1/4", "1/2", "1", "2", "4"))
        diff.set_xlim(0.054, 4.65)
        diff.set_xlabel("记忆负载 P/d")
    axes[0, 0].set_ylim(0.60, 1.025)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0, 0].set_ylabel("比特正确率")
    axes[1, 0].set_ylabel("PDN 相对 Delta\n（百分点）", labelpad=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.074, 0.841),
        ncol=len(METHODS),
        frameon=False,
        columnspacing=2.3,
    )
    fig.text(
        0.082,
        0.055,
        "下排阴影为每组 200 批记忆配对差值的近似 95% 置信区间，虚线为两种方法相等。",
        color="#405560",
    )
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def plot_memory(rows, output):
    plt = setup_plots()
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.5))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.69, wspace=0.27)
    fig.text(0.08, 0.919, "连续写入一条记忆流需要多少内存", fontsize=19, weight="bold")
    fig.text(0.08, 0.852, "FP32，内容维度 m = 32，1 KiB = 1,024 字节", color="#405560")
    panels = (
        ("persistent_write_bytes", "跨次写入必须保留的状态"),
        ("single_stream_cpu_tensor_peak_bytes", "实际写入时的 CPU 张量峰值"),
    )
    for ax, (field, title) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
        for j, method in enumerate(METHODS):
            vals = [r[field] / 1024 for r in rows if r["method"] == method]
            bars = ax.bar(
                np.arange(len(KEY_DIMS)) + (j - (len(METHODS) - 1) / 2) * 0.32,
                vals,
                width=0.29,
                color=COLORS[method],
                label=LABELS[method],
            )
            decimals = 3 if field == "persistent_write_bytes" else 2
            labels = [f"{v:.{decimals}f}".rstrip("0").rstrip(".") for v in vals]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
        ax.set_xticks(np.arange(len(KEY_DIMS)), [f"d = {d}" for d in KEY_DIMS])
        ax.set_ylabel("内存（KiB）")
        ax.grid(axis="y", linewidth=0.6, color="#DEE5E8")
        ax.margins(y=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.073, 0.814),
        ncol=len(METHODS),
        frameon=False,
    )
    fig.text(
        0.08,
        0.107,
        "左图计入权重及继续写入所需的辅助状态。右图用 PyTorch 分配事件测量，每次输入 1 对，连续写入 32 对。",
        color="#405560",
        fontsize=9,
    )
    fig.text(
        0.08,
        0.059,
        "峰值包括临时张量与原版 Delta 的梯度。输入已预分配，Python 进程、运行库及分配器保留空间另计。",
        color="#405560",
        fontsize=9,
    )
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)
