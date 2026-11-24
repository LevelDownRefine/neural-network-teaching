"""在实验一相同配对上对照固定与递减步长 Delta 与作业基线。

所有方法在相同的逐对输入上写一遍，再用相同的完整线索和单一矩阵 W 检索。
写入预算按必须跨样本保留的 FP32 矩阵计。标量、临时张量和评估数据另计。
"""

from itertools import repeat
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from associative_memory.experiment1.run import (
    BASE_SEED,
    KEY_DIMS,
    LEARNING_RATE,
    METHODS,
    METRICS,
    PAIR_COUNTS,
    REPEATS,
    VALUE_DIM,
    measure,
    pair_tensors,
    run_trial,
    sample_pairs,
    write_csv,
)
from associative_memory.models import ModelMemory
from associative_memory.train import DeltaTrainer

from . import OUTPUT_DIR
from .optimized_writes_curves import plot_results

RULES = (*METHODS, "Delta_lr1", "Delta_decay")
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
def evaluate(model, keys, targets):
    model.eval()
    scores = model(q=keys).double().numpy()
    norm = torch.linalg.vector_norm(model.W.double()).item()
    return measure(scores, targets, norm), scores


def run_paired_trial(d, p, repeat_index):
    signs, targets = sample_pairs(d, p, repeat_index)
    keys, values = pair_tensors(signs, targets)
    baselines = run_trial(d, p, repeat_index)

    constant = fit_delta(keys, values, repeat(1.0, p))
    decreasing = fit_delta(keys, values, (d / (d + t) for t in range(p)))
    constant_metrics, _ = evaluate(constant, keys, targets)
    decreasing_metrics, _ = evaluate(decreasing, keys, targets)
    return np.vstack((baselines, constant_metrics, decreasing_metrics))


def compare_condition(d, p, indices, split):
    n = len(indices)
    values = np.empty((n, len(RULES), len(METRICS)))
    for index, seed_index in enumerate(indices):
        values[index] = run_paired_trial(d, p, seed_index)
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
    }
    for i, rule in enumerate(RULES):
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
            "write_passes": 1,
            "weight_updates": 0 if rule == "Zero" else p,
            "weights_dtype": "float32",
            "read_weight_scalars": VALUE_DIM * d,
            "write_extra_matrix_scalars": 0,
            "read_weight_bytes": 4 * VALUE_DIM * d,
            "write_persistent_matrix_bytes": 4 * VALUE_DIM * d,
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
        rows.append(row)
    print(f"{split}   d={d:3d}  P={p:3d}   {n} 次配对重复", flush=True)
    return rows


def main():
    torch.set_num_threads(1)
    rows = []
    for d in KEY_DIMS:
        for p in PAIR_COUNTS:
            rows.extend(compare_condition(d, p, range(REPEATS), "original"))
    for d, p in FRESH_CONDITIONS:
        rows.extend(
            compare_condition(d, p, range(REPEATS, REPEATS + FRESH_REPEATS), "fresh")
        )
    output = OUTPUT_DIR / "optimized_writes"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        staging = Path(tmp)
        write_csv(rows, staging / "comparison.csv")
        plot_results(rows, staging / "recall_vs_load.png")
        for path in sorted(staging.iterdir()):
            path.replace(output / path.name)
    print(output)


if __name__ == "__main__":
    main()
