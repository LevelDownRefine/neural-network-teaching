from .run import KEY_DIMS, PAIR_COUNTS, REPEATS

RIDGE = 0.01


def plot_results(rows, output):
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
            "axes.unicode_minus": False,
        }
    )
    styles = {
        "Hebbian": ("Hebbian 0.5", "#697681", "v", ":"),
        "Delta": ("Delta 0.5", "#24578A", "o", "-"),
        "Delta_lr1": ("Delta 1.0", "#916F3C", "^", "--"),
        "Delta_decay": ("递减步长 Delta", "#13796D", "D", "-"),
        "RLS": ("岭回归 RLS", "#AF4331", "s", "-"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.9), sharey="row")
    fig.subplots_adjust(
        left=0.082, right=0.985, top=0.755, bottom=0.145, wspace=0.18, hspace=0.45
    )
    fig.text(
        0.082, 0.94, "完整线索的单遍检索", fontsize=21, weight="bold", color="#192F3D"
    )
    fig.text(
        0.082,
        0.886,
        f"相同配对与顺序    ·    每组 {REPEATS} 次随机重复    ·    岭正则 λ = {RIDGE}",
        fontsize=10,
        color="#43545F",
    )
    for col, d in enumerate(KEY_DIMS):
        top, bottom = axes[:, col]
        top.set_title(f"d = {d}", fontsize=12, weight="bold", loc="left", pad=6)
        for rule, (label, color, marker, line) in styles.items():
            group = [
                r
                for r in rows
                if r["split"] == "original" and r["d"] == d and r["method"] == rule
            ]
            style = dict(
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.6,
                markersize=4,
                label=label,
            )
            top.plot(
                [r["load"] for r in group],
                [r["bit_accuracy_mean"] for r in group],
                **style,
            )
            bottom.plot(
                [r["P"] for r in group], [r["mse_mean"] for r in group], **style
            )
        for ax in (top, bottom):
            ax.set_xscale("log", base=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#A5B2BA")
            ax.grid(axis="y", linewidth=0.65, color="#DFE5E9")
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=9, length=3)
        loads = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4)
        top.set_xticks(loads, ["1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
        top.set_xlim(0.054, 4.6)
        top.set_xlabel("记忆负载 P/d", fontsize=10)
        bottom.set_yscale("log")
        bottom.set_xticks(PAIR_COUNTS, [str(p) for p in PAIR_COUNTS])
        bottom.set_xlim(6.8, 148)
        bottom.set_xlabel("记忆数量 P", fontsize=10)
    axes[0, 0].set_ylabel("比特正确率")
    axes[0, 0].set_ylim(0.49, 1.025)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1, 0].set_ylabel("连续 MSE（对数轴）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.074, 0.858),
        ncol=5,
        frameon=False,
        fontsize=10,
        columnspacing=1.6,
    )
    fig.text(
        0.082,
        0.044,
        "递减步长 Delta 只增加步数计数。RLS 在写入时另外维护一个 d × d 的 FP32 矩阵。",
        fontsize=10,
        color="#43545F",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
