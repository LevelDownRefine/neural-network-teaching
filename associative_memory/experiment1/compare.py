"""对照已保存的 NumPy 汇总表，检查相同配对上的连续读出和符号变化。"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from associative_memory.recall_benchmark import (
    LEARNING_RATE,
    baseline_summary,
    write_csv,
)

from . import OUTPUT_DIR


def numpy_readouts(key_signs, targets, learning_rate):
    """复算旧版 FP64 读出，供逐比特核对。汇总对照的基准直接读原 CSV。"""
    d = key_signs.shape[1]
    # 旧版存 A = sqrt(d) * W。符号均为整数且 η = 0.5，Hebbian 外积和可精确计算。
    hebbian = learning_rate * (targets.T @ key_signs)
    delta = np.zeros_like(hebbian)
    for s, target in zip(key_signs, targets):
        delta += learning_rate * np.outer(target - (delta @ s) / d, s)

    output = {"Zero": (np.zeros_like(targets), 0.0)}
    for method, weights in (("Hebbian", hebbian), ("Delta", delta)):
        output[method] = (
            key_signs @ weights.T / d,
            np.linalg.norm(weights) / np.sqrt(d),
        )
    return output


@dataclass
class ReadoutAudit:
    bits: int = 0
    flips: int = 0
    flips_at_numpy_zero: int = 0
    numpy_correct: int = 0
    torch_correct: int = 0
    numpy_squared_error: float = 0.0
    numpy_norm_sum: float = 0.0
    max_readout_difference: float = 0.0
    max_numpy_at_flip: float = 0.0
    max_torch_at_flip: float = 0.0

    def add(self, current, reference, targets, reference_norm):
        old_bits = reference >= 0
        new_bits = current >= 0
        positive = targets > 0
        changed = new_bits != old_bits
        self.bits += targets.size
        self.flips += int(np.count_nonzero(changed))
        self.numpy_correct += int(np.count_nonzero(old_bits == positive))
        self.torch_correct += int(np.count_nonzero(new_bits == positive))
        self.numpy_squared_error += float(np.sum((reference - targets) ** 2))
        self.numpy_norm_sum += reference_norm
        self.max_readout_difference = max(
            self.max_readout_difference, np.max(np.abs(current - reference))
        )
        if np.any(changed):
            old_changed, new_changed = reference[changed], current[changed]
            self.flips_at_numpy_zero += int(np.count_nonzero(old_changed == 0))
            self.max_numpy_at_flip = max(
                self.max_numpy_at_flip, float(np.max(np.abs(old_changed)))
            )
            self.max_torch_at_flip = max(
                self.max_torch_at_flip, float(np.max(np.abs(new_changed)))
            )


def compare_summaries(current_rows, baseline_path, audits):
    """按 d、P、方法对齐，并核验比较对象和旧版汇总值。"""
    with baseline_path.open(encoding="utf-8-sig", newline="") as handle:
        baseline = list(csv.DictReader(handle))

    def key(row):
        return int(row["d"]), int(row["P"]), row["method"]

    old_rows = {key(row): row for row in baseline}
    new_keys = {key(row) for row in current_rows}
    if (
        len(old_rows) != len(baseline)
        or len(new_keys) != len(current_rows)
        or old_rows.keys() != new_keys
    ):
        raise ValueError("两版 CSV 的实验组合或方法不一致")

    results = []
    metrics = (
        "bit_accuracy",
        "mse",
        "weight_frobenius",
        "first_quarter_bit_accuracy",
        "last_quarter_bit_accuracy",
    )
    for new in current_rows:
        group = key(new)
        old, audit = old_rows[group], audits[group]
        for field in ("m", "eta", "load", "base_seed", "repeats", "weight_scalars"):
            if float(old[field]) != float(new[field]):
                raise ValueError(f"{group} 的 {field} 与原实验不一致")
        if audit.bits != new["repeats"] * new["P"] * new["m"]:
            raise ValueError(f"{group} 的逐比特核对数目不完整")
        # 独立复算的参照必须同时解释旧表中的正确率、MSE 和权重范数。
        np.testing.assert_allclose(
            [
                audit.numpy_correct / audit.bits,
                audit.numpy_squared_error / audit.bits,
                audit.numpy_norm_sum / new["repeats"],
                audit.torch_correct / audit.bits,
            ],
            [
                float(old["bit_accuracy_mean"]),
                float(old["mse_mean"]),
                float(old["weight_frobenius_mean"]),
                new["bit_accuracy_mean"],
            ],
            rtol=1e-11,
            atol=1e-13,
            err_msg=str(group),
        )
        row = {
            name: new[name]
            for name in ("d", "m", "P", "load", "method", "eta", "repeats", "base_seed")
        }
        row["numpy_weight_bytes"] = int(old["weight_bytes_float64"])
        row["pytorch_weight_bytes"] = new["weight_bytes"]
        for metric in metrics:
            previous, current = float(old[f"{metric}_mean"]), new[f"{metric}_mean"]
            row[f"numpy_{metric}"] = previous
            row[f"pytorch_{metric}"] = current
            if "accuracy" in metric:
                row[f"{metric}_diff_pp"] = 100 * (current - previous)
            else:
                row[f"{metric}_diff"] = current - previous
        row.update(
            bits_compared=audit.bits,
            prediction_flips=audit.flips,
            flips_at_numpy_exact_zero=audit.flips_at_numpy_zero,
            max_abs_readout_diff=audit.max_readout_difference,
            max_abs_numpy_z_at_flip=audit.max_numpy_at_flip,
            max_abs_pytorch_z_at_flip=audit.max_torch_at_flip,
        )
        results.append(row)
    return results


def plot_differences(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

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
    palette = LinearSegmentedColormap.from_list(
        "changes", ["#27627D", "#F6F8F8", "#AB452B"]
    )
    methods = ("Hebbian", "Delta", "Zero")
    groups = [(d, method) for d in sorted({r["d"] for r in rows}) for method in methods]
    counts = sorted({r["P"] for r in rows})
    lookup = {(r["d"], r["method"], r["P"]): r for r in rows}
    specs = (
        ("bit_accuracy_diff_pp", "比特正确率", "百分点"),
        ("mse_diff", "连续读出 MSE", ""),
        ("weight_frobenius_diff", r"权重范数 $\|W\|_F$", ""),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 6.4))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.73, bottom=0.17, wspace=0.12)
    fig.text(
        0.055,
        0.92,
        "同一批记忆上的实现差异",
        fontsize=20,
        weight="bold",
        color="#172837",
    )
    fig.text(
        0.055,
        0.85,
        f"PyTorch FP32 − NumPy FP64    ·    每格为 {rows[0]['repeats']} 次配对重复的均值差",
        fontsize=11,
        color="#415261",
    )
    for col, (ax, (field, title, unit)) in enumerate(zip(axes, specs)):
        data = np.array(
            [[lookup[d, method, p][field] for p in counts] for d, method in groups]
        )
        peak = float(np.max(np.abs(data)))
        exponent = math.floor(math.log10(peak)) if col and peak else 0
        scale = 10.0**exponent
        if exponent:
            unit = rf"$\times\,10^{{{exponent}}}$"
        display = data / scale
        limit = peak / scale if peak else 1.0
        ax.imshow(
            display,
            cmap=palette,
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{title}  {unit}", loc="left", fontsize=12, pad=9, weight="bold")
        ax.set_xticks(range(len(counts)), [str(p) for p in counts])
        ax.set_xlabel("记忆数量 P", fontsize=10)
        labels = [
            f"d={d}    {method if method != 'Zero' else '零矩阵'}"
            for d, method in groups
        ]
        ax.set_yticks(range(len(groups)), labels if col == 0 else [""] * len(groups))
        ax.tick_params(length=0, labelsize=9, pad=6)
        for side in ax.spines.values():
            side.set_visible(False)
        for divider in range(len(methods), len(groups), len(methods)):
            ax.axhline(divider - 0.5, color="#A3AFB7", linewidth=0.8)
        decimals = 3 if col == 0 else 2
        for (i, j), value in np.ndenumerate(display):
            label = f"{value:+.{decimals}f}" if round(float(value), decimals) else "0"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > 0.65 * limit else "#213340",
            )
    fig.text(
        0.055,
        0.054,
        "蓝色表示 PyTorch 数值较低，橙色表示较高。各列独立着色。格内数值经四舍五入，完整精度见对照 CSV。",
        fontsize=9.5,
        color="#415261",
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    audits = defaultdict(ReadoutAudit)

    def audit_readouts(d, p, signs, targets, readouts):
        references = numpy_readouts(signs, targets, LEARNING_RATE)
        for method, (scores, _) in readouts.items():
            reference, norm = references[method]
            audits[d, p, method].add(scores, reference, targets, norm)

    rows = baseline_summary(observe=audit_readouts)
    reference = Path(__file__).with_name("reference") / "numpy_fp64.csv"
    comparisons = compare_summaries(rows, reference, audits)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_DIR) as temp:
        staging = Path(temp)
        write_csv(comparisons, staging / "experiment1_comparison.csv")
        plot_differences(comparisons, staging / "experiment1_comparison.png")
        for path in sorted(staging.iterdir()):
            destination = OUTPUT_DIR / path.name
            path.replace(destination)
            print(destination)


if __name__ == "__main__":
    main()
