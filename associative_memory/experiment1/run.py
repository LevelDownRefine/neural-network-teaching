"""完整线索检索的配对数据、三种基线与评估指标。"""

import csv
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
from .curves import plot_curves

KEY_DIMS = (32, 64, 128)
PAIR_COUNTS = (8, 16, 32, 64, 128)
VALUE_DIM = 32
LEARNING_RATE = 0.5
REPEATS = 200
BASE_SEED = 20261120
TRAINERS = {"Hebbian": HebbianTrainer, "Delta": DeltaTrainer, "Zero": ZeroTrainer}
METHODS = tuple(TRAINERS)
METRICS = (
    "bit_accuracy",
    "mse",
    "weight_frobenius",
    "first_quarter_bit_accuracy",
    "last_quarter_bit_accuracy",
)


def sample_pairs(d, p, repeat):
    """返回随机双极键的符号和目标内容，沿用原 NumPy 实验的抽样顺序。"""
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, d, p, repeat]))
    key_bits = rng.integers(0, 2, size=(p, d), dtype=np.int8)
    value_bits = rng.integers(0, 2, size=(p, VALUE_DIM), dtype=np.int8)
    return (2 * key_bits - 1).astype(np.float64), (2 * value_bits - 1).astype(
        np.float64
    )


def pair_tensors(signs, targets):
    keys = (signs / np.sqrt(signs.shape[1])).astype(np.float32)
    return torch.from_numpy(keys), torch.from_numpy(targets.astype(np.float32))


def measure(scores, targets, weight_norm):
    correct = np.where(scores >= 0, 1, -1) == targets
    quarter = max(1, len(targets) // 4)
    return (
        np.mean(correct),
        np.mean((scores - targets) ** 2),
        weight_norm,
        np.mean(correct[:quarter]),
        np.mean(correct[-quarter:]),
    )


def run_trial(d, p, repeat):
    """每条配对写一次，最后用完整线索读出。可选回调接收同批数据和各方法的连续读出。"""
    signs, targets = sample_pairs(d, p, repeat)
    keys, values = pair_tensors(signs, targets)
    readouts = {}
    for method, trainer_class in TRAINERS.items():
        model = ModelMemory(d=d, m=VALUE_DIM, device="cpu")
        trainer = (
            trainer_class(model)
            if method == "Zero"
            else trainer_class(model, lr=LEARNING_RATE)
        )
        train_epoch(trainer, keys, values, range(p))
        model.eval()
        with torch.no_grad():
            scores = model(q=keys).double().numpy()
            norm = torch.linalg.vector_norm(model.W.double()).item()
        readouts[method] = scores, norm
    return np.asarray(
        [measure(scores, targets, norm) for scores, norm in readouts.values()]
    )


def baseline_summary():
    rows = []
    for d in KEY_DIMS:
        weight = ModelMemory(d=d, m=VALUE_DIM).W
        for p in PAIR_COUNTS:
            runs = np.stack([run_trial(d, p, i) for i in range(REPEATS)])
            means = runs.mean(axis=0)
            deviations = runs.std(axis=0, ddof=1)
            for i, method in enumerate(METHODS):
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
                for j, metric in enumerate(METRICS):
                    row[f"{metric}_mean"] = means[i, j]
                    row[f"{metric}_sd"] = deviations[i, j]
                rows.append(row)
            print(f"d={d:3d}   P={p:3d}   已完成 {REPEATS} 次重复", flush=True)
    return rows


def write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    rows = baseline_summary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_DIR) as temp:
        staging = Path(temp)
        write_csv(rows, staging / "experiment1_summary.csv")
        plot_curves(
            rows,
            staging / "experiment1_curves.png",
            f"PyTorch FP32    ·    内容维数 m = {VALUE_DIM}    ·    学习率 η = {LEARNING_RATE}    ·    每组 {REPEATS} 次独立实验",
        )
        for path in sorted(staging.iterdir()):
            destination = OUTPUT_DIR / path.name
            path.replace(destination)
            print(destination)


if __name__ == "__main__":
    main()
