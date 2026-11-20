"""Experiment 1: one-pass capacity of key–value associative memory.

Run from repository root: python -m associative_memory.experiment1
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import generate_pairs
from .models import ModelDelta, ModelHebbian, ModelZero

MODELS = {"Hebbian": ModelHebbian, "Delta": ModelDelta, "Zero": ModelZero}


def run(seed=123, dimensions=(32, 64, 128), counts=(8, 16, 32, 64, 128), m=32, eta=0.5):
    """Each condition restarts its seed; methods share pairs and order."""
    rows = []
    for d in dimensions:
        for p in counts:
            rng = np.random.default_rng(seed)
            keys, values, order = generate_pairs(rng, p, d, m)
            for name, cls in MODELS.items():
                model = cls(d=d, m=m, eta=eta)
                for i in order:
                    model.feed_data({"k": keys[i], "v": values[i]})
                    model.optimize_parameters()
                model.feed_data({"q": keys})
                model.test()
                result = model.current_results()
                rows.append(
                    {
                        "seed": seed,
                        "d": d,
                        "m": m,
                        "P": p,
                        "load": p / d,
                        "method": name,
                        "accuracy": float(np.mean(result["v_hat"] == values)),
                        "mse": float(np.mean((result["z"] - values) ** 2)),
                        "weight_norm": float(np.linalg.norm(model.W, ord="fro")),
                    }
                )
    return rows


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output, seed):
    colors = {32: "tab:blue", 64: "tab:orange", 128: "tab:green"}
    styles = {"Hebbian": "-", "Delta": "--", "Zero": ":"}
    for metric, xkey, xlabel, ylabel in (
        ("accuracy", "load", "Memory load P/d", "Bit accuracy"),
        ("mse", "P", "Number of memories P", "Continuous-output MSE"),
        ("weight_norm", "P", "Number of memories P", "Frobenius norm of W"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for d in sorted({r["d"] for r in rows}):
            for method in MODELS:
                points = sorted(
                    [r for r in rows if r["d"] == d and r["method"] == method],
                    key=lambda r: r[xkey],
                )
                x = np.array([r[xkey] for r in points])
                y = np.array([r[metric] for r in points])
                ax.plot(
                    x,
                    y,
                    styles[method],
                    marker="o",
                    markersize=3,
                    color=colors.get(d),
                    label=f"{method}, d={d}",
                )
        ax.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"One-pass associative memory (seed={seed})",
        )
        if metric == "accuracy":
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output" / "experiment1"
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = run(args.seed)
    save_csv(args.output / "metrics.csv", rows)
    config = {
        "seed": args.seed,
        "d": [32, 64, 128],
        "P": [8, 16, 32, 64, 128],
        "m": 32,
        "eta": 0.5,
        "epochs": 1,
        "dtype": "float64",
        "random_scheme": "numpy.default_rng(seed); reset per condition",
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    plot(rows, args.output, args.seed)
    print(f"Saved {len(rows)} measurements and 3 plots to {args.output.resolve()}")
    for r in rows:
        if r["d"] == 64:
            print(
                f"d=64 P={r['P']:3} {r['method']:7} accuracy={r['accuracy']:.3f} MSE={r['mse']:.3f} norm={r['weight_norm']:.3f}"
            )


if __name__ == "__main__":
    main()
