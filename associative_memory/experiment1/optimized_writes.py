"""在实验一相同配对上对照固定与递减步长 Delta、RLS 和必做基线。

所有方法在相同的逐对输入上写一遍，再用相同的完整线索和单一矩阵 W 检索。
写入预算按必须跨样本保留的 FP32 矩阵计。标量、临时张量和评估数据另计。
"""

import csv
from itertools import repeat
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from associative_memory.models import ModelMemory
from associative_memory.train import DeltaTrainer

from . import OUTPUT_DIR, run
from .run import (
    BASE_SEED,
    KEY_DIMS,
    LEARNING_RATE,
    METHODS,
    METRICS,
    PAIR_COUNTS,
    REPEATS,
    VALUE_DIM,
    measure,
    random_signs,
    run_trial,
    write_csv,
)

RIDGE = 0.01
RULES = (*METHODS, "Delta_lr1", "Delta_decay", "RLS")
FRESH_REPEATS = 100
FRESH_CONDITIONS = ((64, 32), (32, 128))


def fit_delta(keys, values, rates):
    model = ModelMemory(d=keys.shape[1], m=values.shape[1])
    trainer = DeltaTrainer(model, lr=1.0)
    for key, value, rate in zip(keys, values, rates, strict=True):
        trainer.optimizer.param_groups[0]["lr"] = rate
        trainer.step({"q": key}, value)
    return model


@torch.no_grad()
def fit_rls(keys, values, ridge=RIDGE):
    """递归岭回归，无遗忘。逆矩阵从 I/λ 开始，更新只使用当前的键和值。"""
    d = keys.shape[1]
    model = ModelMemory(d=d, m=values.shape[1])
    inverse_gram = torch.eye(d, dtype=keys.dtype) / ridge
    for key, value in zip(keys, values, strict=True):
        direction = inverse_gram @ key
        inverse_denom = (1.0 + torch.dot(key, direction)).reciprocal()
        residual = value - model(q=key)
        model.W.add_(torch.outer(residual, direction * inverse_denom))
        inverse_gram.sub_(torch.outer(direction, direction) * inverse_denom)
    return model


@torch.no_grad()
def evaluate(model, keys, targets):
    model.eval()
    scores = model(q=keys).double().numpy()
    norm = torch.linalg.vector_norm(model.W.double()).item()
    return measure(scores, targets, norm), scores


def run_paired_trial(d, p, repeat_index):
    # 原脚本按这些种子分开生成每组样本。三条基线也用原入口重新运行。
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, d, p, repeat_index]))
    signs = random_signs(rng, (p, d))
    targets = random_signs(rng, (p, VALUE_DIM))
    keys = torch.from_numpy((signs / np.sqrt(d)).astype(np.float32))
    values = torch.from_numpy(targets.astype(np.float32))
    baselines = run_trial(d, p, repeat_index)

    constant = fit_delta(keys, values, repeat(1.0, p))
    decreasing = fit_delta(keys, values, (d / (d + t) for t in range(p)))
    rls = fit_rls(keys, values)
    constant_metrics, _ = evaluate(constant, keys, targets)
    decreasing_metrics, _ = evaluate(decreasing, keys, targets)
    rls_metrics, rls_scores = evaluate(rls, keys, targets)

    # 只在写完后以 FP64 批量求解同一岭回归问题，独立核验在线实现。
    x = keys.double().numpy()
    gram = x.T @ x
    gram[np.diag_indices(d)] += RIDGE
    solution = np.linalg.solve(gram, x.T @ targets)
    batch_scores = x @ solution
    np.testing.assert_allclose(rls_scores, batch_scores, atol=4e-4, rtol=2e-4)
    discrepancy = np.max(np.abs(rls_scores - batch_scores))
    all_metrics = np.vstack(
        (baselines, constant_metrics, decreasing_metrics, rls_metrics)
    )
    return all_metrics, discrepancy


def compare_condition(d, p, indices, split):
    n = len(indices)
    values = np.empty((n, len(RULES), len(METRICS)))
    ridge_error = 0.0
    for index, seed_index in enumerate(indices):
        values[index], discrepancy = run_paired_trial(d, p, seed_index)
        ridge_error = max(ridge_error, discrepancy)
    avg = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    base = values[:, RULES.index("Delta"), :]
    rows = []
    rates = {
        "Hebbian": str(LEARNING_RATE),
        "Delta": str(LEARNING_RATE),
        "Zero": "",
        "Delta_lr1": "1",
        "Delta_decay": "d/(d+t-1), t=1..P",
        "RLS": "",
    }
    for i, rule in enumerate(RULES):
        extra = d * d if rule == "RLS" else 0
        row = {
            "split": split,
            "d": d,
            "m": VALUE_DIM,
            "P": p,
            "load": p / d,
            "method": rule,
            "repeats": n,
            "base_seed": BASE_SEED,
            "first_repeat_index": indices.start,
            "last_repeat_index": indices.stop - 1,
            "learning_rate": rates[rule],
            "ridge_lambda": RIDGE if rule == "RLS" else "",
            "write_passes": 1,
            "weight_updates": 0 if rule == "Zero" else p,
            "weights_dtype": "float32",
            "read_weight_scalars": VALUE_DIM * d,
            "write_extra_matrix_scalars": extra,
            "read_weight_bytes": 4 * VALUE_DIM * d,
            "write_persistent_matrix_bytes": 4 * (VALUE_DIM * d + extra),
        }
        for j, metric in enumerate(METRICS):
            row[f"{metric}_mean"] = avg[i, j]
            row[f"{metric}_sd"] = sd[i, j]
        # 配对比较的统计单位是一批随机记忆。正值分别表示准确率提高或 MSE 下降。
        gains = {
            "accuracy_gain_vs_delta_pp": 100 * (values[:, i, 0] - base[:, 0]),
            "mse_reduction_vs_delta": base[:, 1] - values[:, i, 1],
        }
        for label, paired in gains.items():
            center = paired.mean()
            half_width = 1.96 * paired.std(ddof=1) / np.sqrt(n)
            row[label] = center
            row[f"{label}_ci95_low"] = center - half_width
            row[f"{label}_ci95_high"] = center + half_width
        row["max_rls_vs_batch_abs_score_error"] = ridge_error if rule == "RLS" else ""
        rows.append(row)
    print(f"{split}   d={d:3d}  P={p:3d}   {n} 次配对重复", flush=True)
    return rows


def verify_baselines(rows, source):
    with source.open(encoding="utf-8-sig", newline="") as file:
        old = {(int(r["d"]), int(r["P"]), r["method"]): r for r in csv.DictReader(file)}
    recovered = [r for r in rows if r["split"] == "original" and r["method"] in METHODS]
    if set(old) != {(r["d"], r["P"], r["method"]) for r in recovered}:
        raise ValueError("基线数据与原主实验的条件不一致")
    for r in recovered:
        archived = old[r["d"], r["P"], r["method"]]
        for metric in METRICS:
            for suffix in ("mean", "sd"):
                name = f"{metric}_{suffix}"
                np.testing.assert_allclose(
                    r[name], float(archived[name]), rtol=1e-11, atol=1e-13
                )


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


def main():
    torch.set_num_threads(1)
    root = OUTPUT_DIR
    if not (root / "experiment1_summary.csv").is_file():
        run.main()
    rows = []
    for d in KEY_DIMS:
        for p in PAIR_COUNTS:
            rows.extend(compare_condition(d, p, range(REPEATS), "original"))
    verify_baselines(rows, root / "experiment1_summary.csv")
    for d, p in FRESH_CONDITIONS:
        rows.extend(
            compare_condition(d, p, range(REPEATS, REPEATS + FRESH_REPEATS), "fresh")
        )
    output = root / "optimized_writes"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        staging = Path(tmp)
        write_csv(rows, staging / "comparison.csv")
        plot_results(rows, staging / "recall_vs_load.png")
        for path in sorted(staging.iterdir()):
            path.replace(output / path.name)
    print(output)


if __name__ == "__main__":
    main()
