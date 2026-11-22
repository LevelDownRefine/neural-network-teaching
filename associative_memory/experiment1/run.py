"""在同一批随机记忆上运行 PyTorch 主实验，并与归档的 NumPy FP64 结果对照。"""

import csv
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from associative_memory.models import ModelMemory
from associative_memory.train import (
    DeltaTrainer,
    HebbianTrainer,
    ZeroTrainer,
    train_epoch,
)

from . import OUTPUT_DIR
from .compare import ReadoutAudit, compare_summaries, numpy_readouts, plot_differences

KEY_DIMS = (32, 64, 128)
PAIR_COUNTS = (8, 16, 32, 64, 128)
VALUE_DIM = 32
LEARNING_RATE = 0.5
REPEATS = 200
BASE_SEED = 20261120
TRAINERS = {"Hebbian": HebbianTrainer, "Delta": DeltaTrainer, "Zero": ZeroTrainer}
METHODS = tuple(TRAINERS)
PLOT_METRICS = ("bit_accuracy", "mse", "weight_frobenius")
METRICS = (*PLOT_METRICS, "first_quarter_bit_accuracy", "last_quarter_bit_accuracy")


def random_signs(rng, shape):
    bits = rng.integers(0, 2, size=shape, dtype=np.int8)
    return (2 * bits - 1).astype(np.float64)


def measure(scores, targets, weight_norm):
    """用实际的 FP32 读出判定符号，以 FP64 汇总各项指标。"""
    predicted_bits = np.where(scores >= 0, 1, -1)
    correct = predicted_bits == targets
    # 按原始写入顺序分组，观察最早和最晚写入的四分之一是否同样容易找回。
    quarter = max(1, len(targets) // 4)
    return (
        np.mean(correct),
        np.mean((scores - targets) ** 2),
        weight_norm,
        np.mean(correct[:quarter]),
        np.mean(correct[-quarter:]),
    )


def run_trial(d, p, repeat, *, audit=None):
    """按 NumPy 旧版的随机数流生成数据，三个上游 Trainer 共享同一写入顺序。"""
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, d, p, repeat]))
    key_signs = random_signs(rng, (p, d))
    targets = random_signs(rng, (p, VALUE_DIM))
    keys = torch.from_numpy((key_signs / np.sqrt(d)).astype(np.float32))
    values = torch.from_numpy(targets.astype(np.float32))
    order = range(p)
    if audit is not None:
        reference = numpy_readouts(key_signs, targets, LEARNING_RATE)

    metrics = []
    for method, trainer_class in TRAINERS.items():
        model = ModelMemory(d=d, m=VALUE_DIM, device="cpu")
        trainer = (
            trainer_class(model)
            if method == "Zero"
            else trainer_class(model, lr=LEARNING_RATE)
        )
        train_epoch(trainer, keys, values, order)
        model.eval()
        with torch.no_grad():
            scores = model(q=keys).double().numpy()
            weight_norm = torch.linalg.vector_norm(model.W.double()).item()
        metrics.append(measure(scores, targets, weight_norm))
        if audit is not None:
            reference_scores, reference_norm = reference[method]
            audit[d, p, method].add(scores, reference_scores, targets, reference_norm)
    return np.asarray(metrics)


def run_experiment(*, audit=None):
    rows = []
    for d in KEY_DIMS:
        weight = ModelMemory(d=d, m=VALUE_DIM).W
        for p in PAIR_COUNTS:
            runs = np.stack(
                [run_trial(d, p, repeat, audit=audit) for repeat in range(REPEATS)]
            )
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
                    "weight_scalars": weight.numel(),
                    "weight_dtype": str(weight.dtype).removeprefix("torch."),
                    "weight_bytes": weight.numel() * weight.element_size(),
                    "device": weight.device.type,
                    "torch_threads": torch.get_num_threads(),
                    "torch_version": str(torch.__version__),
                    "numpy_version": np.__version__,
                }
                for metric_index, metric in enumerate(METRICS):
                    row[f"{metric}_mean"] = means[method_index, metric_index]
                    row[f"{metric}_sd"] = deviations[method_index, metric_index]
                rows.append(row)
            print(f"d={d:3d}   P={p:3d}   已完成 {REPEATS} 次重复", flush=True)
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
        f"PyTorch FP32    ·    内容维数 m = {VALUE_DIM}    ·    学习率 η = {LEARNING_RATE}    ·    每组 {REPEATS} 次独立实验",
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


def write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = Path(__file__).with_name("reference") / "numpy_fp64.csv"
    audit = defaultdict(ReadoutAudit)
    rows = run_experiment(audit=audit)
    comparisons = compare_summaries(rows, baseline, audit)
    with TemporaryDirectory(dir=output_dir, prefix=".experiment1-") as temp:
        staging = Path(temp)
        write_csv(rows, staging / "experiment1_summary.csv")
        write_csv(comparisons, staging / "experiment1_comparison.csv")
        plot_curves(rows, staging / "experiment1_curves.png")
        plot_differences(comparisons, staging / "experiment1_comparison.png")
        for file in sorted(staging.iterdir()):
            destination = output_dir / file.name
            file.replace(destination)
            print(destination)


if __name__ == "__main__":
    main()
