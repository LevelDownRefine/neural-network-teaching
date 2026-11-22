"""实验一的三列评分曲线，供 PyTorch 和 NumPy 共用。"""

import numpy as np

PLOT_METRICS = ("bit_accuracy", "mse", "weight_frobenius")


def plot_curves(rows, path, subtitle):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(12.1, 8.6), sharex="col", sharey="col")
    fig.subplots_adjust(
        left=0.103, right=0.985, bottom=0.115, top=0.83, wspace=0.22, hspace=0.22
    )
    fig.text(
        0.103,
        0.942,
        "一次写入后的联想记忆",
        fontsize=21,
        fontweight="bold",
        color="#172837",
    )
    fig.text(
        0.103,
        0.900,
        subtitle,
        fontsize=10.5,
        color="#415261",
    )
    titles = ("比特正确率", "连续读出 MSE", r"权重范数 $\|W\|_F$")
    styles = {
        "Hebbian": ("#146580", "o", "-", "Hebbian"),
        "Delta": ("#B64829", "s", "-", "Delta"),
        "Zero": ("#616A78", "^", "--", "零矩阵"),
    }
    dimensions = sorted({row["d"] for row in rows})
    counts = sorted({row["P"] for row in rows})
    for row_index, d in enumerate(dimensions):
        axes[row_index, 0].set_ylabel(
            f"d = {d}",
            rotation=0,
            labelpad=23,
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        for col_index, metric in enumerate(PLOT_METRICS):
            ax = axes[row_index, col_index]
            ax.set_xscale("log", base=2)
            ax.spines[["top", "right"]].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#B2BCC4")
            ax.tick_params(axis="both", labelsize=9, length=3, color="#8996A1")
            ax.grid(axis="y", color="#DFE5E9", linewidth=0.7)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(
                    titles[col_index],
                    loc="left",
                    pad=10,
                    fontweight="bold",
                    fontsize=12,
                )
            for method, (color, marker, line, label) in styles.items():
                group = [r for r in rows if r["d"] == d and r["method"] == method]
                x = [r["load" if col_index == 0 else "P"] for r in group]
                y = np.array([r[f"{metric}_mean"] for r in group])
                sd = np.array([r[f"{metric}_sd"] for r in group])
                lower = np.maximum(0, y - sd)
                upper = np.minimum(1, y + sd) if col_index == 0 else y + sd
                ax.fill_between(x, lower, upper, color=color, alpha=0.11, linewidth=0)
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    linestyle=line,
                    lw=1.7,
                    ms=4,
                    label=label,
                )

    axes[0, 0].set_ylim(0.44, 1.025)
    axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 2].set_ylim(bottom=-0.8)
    loads = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
    axes[-1, 0].set_xticks(loads, ["1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
    axes[-1, 0].set_xlim(0.054, 4.65)
    axes[-1, 0].set_xlabel("记忆负载 P/d")
    for ax in axes[-1, 1:]:
        ax.set_xticks(counts, [str(p) for p in counts])
        ax.set_xlim(6.8, 149)
        ax.set_xlabel("记忆数量 P")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.986, 0.96),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.text(
        0.103,
        0.025,
        "点为重复实验的均值，阴影为均值上下一个样本标准差。",
        color="#415261",
        fontsize=9.5,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
