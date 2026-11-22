"""使用独立的 NumPy FP64 实现复算基线，并输出比特正确率、MSE 和连接矩阵范数。"""

import csv

import numpy as np

from . import OUTPUT_DIR
from .curves import PLOT_METRICS, plot_curves

KEY_DIMS = (32, 64, 128)
PAIR_COUNTS = (8, 16, 32, 64, 128)
VALUE_DIM = 32
LEARNING_RATE = 0.5
REPEATS = 200
BASE_SEED = 20261120
METHODS = ("Hebbian", "Delta", "Zero")
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
    plot_curves(
        rows,
        figure_path,
        f"内容维数 m = {VALUE_DIM}     学习率 η = {LEARNING_RATE}     每组 {REPEATS} 次独立实验",
    )
    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
