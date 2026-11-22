"""使用独立的 NumPy FP64 实现复算基线，并输出比特正确率、MSE 和连接矩阵范数。"""

import csv

import numpy as np

from . import OUTPUT_DIR

KEY_DIMS = (32, 64, 128)
PAIR_COUNTS = (8, 16, 32, 64, 128)
VALUE_DIM = 32
LEARNING_RATE = 0.5
REPEATS = 200
BASE_SEED = 20261120
METHODS = ("Hebbian", "Delta", "Zero")
PLOT_METRICS = ("bit_accuracy", "mse", "weight_frobenius")
METRICS = (*PLOT_METRICS, "first_quarter_bit_accuracy", "last_quarter_bit_accuracy")


def random_signs(rng, shape):
    bits = rng.integers(0, 2, size=shape, dtype=np.int8)
    return (2 * bits - 1).astype(np.float64)


def measure(scaled_weights, key_signs, targets):
    """以 z = Wk 计算连续 MSE，以 z >= 0 判定内容比特。"""
    d = key_signs.shape[1]
    scores = (key_signs @ scaled_weights.T) / d
    predicted_bits = np.where(scores >= 0, 1, -1)
    correct = predicted_bits == targets
    # 按原始写入顺序分组，观察最早和最晚写入的四分之一是否同样容易找回。
    quarter = max(1, len(targets) // 4)
    return (
        np.mean(correct),
        np.mean((scores - targets) ** 2),
        np.linalg.norm(scaled_weights, ord="fro") / np.sqrt(d),
        np.mean(correct[:quarter]),
        np.mean(correct[-quarter:]),
    )


def run_trial(d, p, repeat):
    """同一组内三种方法共享配对和顺序。不同 (d, p, repeat) 独立抽样。"""
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, d, p, repeat]))
    key_signs = random_signs(rng, (p, d))
    targets = random_signs(rng, (p, VALUE_DIM))

    # 用 s = sqrt(d) * k 和 A = sqrt(d) * W 表示题面的 k、W。
    # 此时 Wk = A @ s / d，两条写入规则均严格等价于原式。
    # 对本实验的 Hebbian 规则，这样可准确保留数学上等于零的读出。
    hebbian = np.zeros((VALUE_DIM, d), dtype=np.float64)
    delta = np.zeros_like(hebbian)
    zero = np.zeros_like(hebbian)
    for s, target in zip(key_signs, targets):
        hebbian += LEARNING_RATE * np.outer(target, s)
        prediction = (delta @ s) / d
        delta += LEARNING_RATE * np.outer(target - prediction, s)

    return np.array([measure(w, key_signs, targets) for w in (hebbian, delta, zero)])


def run_experiment():
    rows = []
    for d in KEY_DIMS:
        for p in PAIR_COUNTS:
            runs = np.stack([run_trial(d, p, repeat) for repeat in range(REPEATS)])
            means = runs.mean(axis=0)
            deviations = runs.std(axis=0, ddof=1)
            for method_index, method in enumerate(METHODS):
                row = {
                    "d": d,
                    "m": VALUE_DIM,
                    "P": p,
                    "load": p / d,
                    "method": method,
                    "eta": LEARNING_RATE,
                    "repeats": REPEATS,
                    "base_seed": BASE_SEED,
                    "weight_scalars": VALUE_DIM * d,
                    "weight_bytes_float64": VALUE_DIM
                    * d
                    * np.dtype(np.float64).itemsize,
                }
                for metric_index, metric in enumerate(METRICS):
                    row[f"{metric}_mean"] = means[method_index, metric_index]
                    row[f"{metric}_sd"] = deviations[method_index, metric_index]
                rows.append(row)
    return rows


def plot_curves(rows, path):
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
        f"内容维数 m = {VALUE_DIM}     学习率 η = {LEARNING_RATE}     每组 {REPEATS} 次独立实验",
        fontsize=10.5,
        color="#415261",
    )
    titles = ("比特正确率", "连续读出 MSE", r"权重范数 $\|W\|_F$")
    styles = {
        "Hebbian": ("#146580", "o", "-", "Hebbian"),
        "Delta": ("#B64829", "s", "-", "Delta"),
        "Zero": ("#616A78", "^", "--", "零矩阵"),
    }
    for row_index, d in enumerate(KEY_DIMS):
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
        ax.set_xticks(PAIR_COUNTS, [str(p) for p in PAIR_COUNTS])
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


def main():
    output_dir = OUTPUT_DIR / "numpy_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_experiment()
    csv_path = output_dir / "experiment1_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure_path = output_dir / "experiment1_curves.png"
    plot_curves(rows, figure_path)
    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
