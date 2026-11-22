"""实验一：比较 Hebbian、Delta 和零矩阵的一次写入检索。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from associative_memory.recall_benchmark import (
    LEARNING_RATE,
    REPEATS,
    VALUE_DIM,
    baseline_summary,
    write_csv,
)

from . import OUTPUT_DIR
from .curves import plot_curves


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
