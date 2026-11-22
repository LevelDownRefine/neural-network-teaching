"""Homework 3, option C: derive and measure Hebbian crosstalk.

Run from the repository root: python -m associative_memory.option_3.run
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from associative_memory.data import generate_pairs
from associative_memory.models import ModelMemory
from associative_memory.train import HebbianTrainer, train_epoch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = tuple(range(1000, 1080))
LOADS = (8, 16, 32, 64, 128)
DIMENSIONS = (32, 64, 128)
ETA = 0.5
FLIP_PROBABILITY = 0.25


def correlated_keys(rng, count, dimension):
    """Independent flips around one random prototype, with pairwise mean overlap 0.25."""
    prototype = rng.choice((-1.0, 1.0), dimension)
    flips = rng.random((count, dimension)) < FLIP_PROBABILITY
    return prototype[None, :] * np.where(flips, -1.0, 1.0) / np.sqrt(dimension)


def evaluate(seed, count, dimension, eta, condition):
    rng = np.random.default_rng(seed)
    if condition == "independent":
        keys, values, order = generate_pairs(rng, count, dimension, 32)
    else:
        keys = correlated_keys(rng, count, dimension)
        values = rng.choice((-1.0, 1.0), (count, 32))
        order = rng.permutation(count)

    # Single-pass Hebbian SGD is exactly the closed form below. The common model
    # and trainer are checked separately to retain fast, deterministic sweeps.
    weights = eta * values.T @ keys
    output = keys @ weights.T
    target = eta * values
    crosstalk = output - target
    overlaps = keys @ keys.T
    np.fill_diagonal(overlaps, 0)
    theoretical_conditional_variance = eta**2 * np.mean(np.sum(overlaps**2, axis=1))
    rho = (1 - 2 * FLIP_PROBABILITY) ** 2
    predicted_variance = eta**2 * (count - 1) * (
        1 / dimension if condition == "independent"
        else rho**2 + (1 - rho**2) / dimension
    )
    return {
        "seed": seed, "condition": condition, "P": count, "d": dimension,
        "eta": eta, "crosstalk_mean": float(crosstalk.mean()),
        "crosstalk_second_moment": float(np.mean(crosstalk**2)),
        "conditional_variance": float(theoretical_conditional_variance),
        "predicted_variance": float(predicted_variance),
        "readout_mse_to_value": float(np.mean((output - values)**2)),
        "bit_error_rate": float(np.mean(np.where(output >= 0, 1, -1) != values)),
        "mean_offdiagonal_overlap": float(overlaps.sum() / (count * (count - 1))),
    }


def check_repository_implementation():
    """Verify that the analytic sweep matches the current PyTorch entry point."""
    rng = np.random.default_rng(2026)
    keys, values, order = generate_pairs(rng, 32, 64, 32)
    model = ModelMemory(d=64, m=32)
    trainer = HebbianTrainer(model, lr=ETA)
    train_epoch(trainer, torch.tensor(keys, dtype=torch.float32),
                torch.tensor(values, dtype=torch.float32), order)
    expected = ETA * values.T @ keys
    np.testing.assert_allclose(model.W.detach().numpy(), expected, atol=2e-6)
    with torch.no_grad():
        actual = model(q=torch.tensor(keys, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(actual, keys @ expected.T, atol=3e-6)


def grouped(rows):
    groups = {}
    for row in rows:
        key = (row["condition"], row["P"], row["d"], row["eta"])
        groups.setdefault(key, []).append(row)
    summary = []
    measures = ("crosstalk_mean", "crosstalk_second_moment", "conditional_variance",
                "readout_mse_to_value", "bit_error_rate", "mean_offdiagonal_overlap")
    for (condition, count, dimension, eta), sample in sorted(groups.items()):
        item = {"condition": condition, "P": count, "d": dimension, "eta": eta,
                "n_seeds": len(sample), "predicted_variance": sample[0]["predicted_variance"]}
        for measure in measures:
            values = [row[measure] for row in sample]
            item[measure + "_mean"] = float(np.mean(values))
            item[measure + "_sd"] = float(np.std(values, ddof=1))
        summary.append(item)
    return summary


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for dimension in DIMENSIONS:
        sample = [r for r in summary if r["condition"] == "independent"
                  and r["d"] == dimension and r["eta"] == ETA]
        axes[0].plot(LOADS, [r["crosstalk_second_moment_mean"] for r in sample],
                     "o-", label=f"d={dimension}, measured")
        axes[0].plot(LOADS, [r["predicted_variance"] for r in sample], "--",
                     label=f"d={dimension}, theory")
        axes[1].plot(LOADS, [r["bit_error_rate_mean"] for r in sample], "o-",
                     label=f"d={dimension}")
    axes[0].set(ylabel="Crosstalk second moment", xlabel="Stored pairs P")
    axes[1].set(ylabel="Bit error rate", xlabel="Stored pairs P")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(RESULTS / "independent_load_dimension.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for condition, label in (("independent", "Independent"), ("correlated", "Correlated cues")):
        sample = [r for r in summary if r["condition"] == condition and r["d"] == 64
                  and r["eta"] == ETA]
        axes[0].plot(LOADS, [r["crosstalk_second_moment_mean"] for r in sample],
                     "o-", label=f"{label}, measured")
        axes[0].plot(LOADS, [r["predicted_variance"] for r in sample], "--",
                     label=f"{label}, theory")
        axes[1].plot(LOADS, [r["bit_error_rate_mean"] for r in sample], "o-",
                     label=label)
    axes[0].set(ylabel="Crosstalk second moment", xlabel="Stored pairs P")
    axes[1].set(ylabel="Bit error rate", xlabel="Stored pairs P")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(RESULTS / "correlated_cues.png", dpi=180)
    plt.close(fig)

    sample = [r for r in summary if r["condition"] == "independent"
              and r["P"] == 32 and r["d"] == 64]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot([r["eta"] for r in sample],
                 [r["crosstalk_second_moment_mean"] for r in sample], "o-", label="Measured")
    axes[0].plot([r["eta"] for r in sample],
                 [r["predicted_variance"] for r in sample], "--", label="Theory")
    axes[0].set(xlabel="Learning rate eta", ylabel="Crosstalk second moment")
    axes[0].legend()
    axes[1].plot([r["eta"] for r in sample], [r["bit_error_rate_mean"] for r in sample], "o-")
    axes[1].set(xlabel="Learning rate eta", ylabel="Bit error rate", ylim=(0, 0.12))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(RESULTS / "learning_rate.png", dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    check_repository_implementation()
    rows = []
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            for count in LOADS:
                rows.append(evaluate(seed, count, dimension, ETA, "independent"))
        for count in LOADS:
            rows.append(evaluate(seed, count, 64, ETA, "correlated"))
        for eta in (0.25, 1.0):
            rows.append(evaluate(seed, 32, 64, eta, "independent"))
    summary = grouped(rows)
    save_csv(RESULTS / "individual_runs.csv", rows)
    save_csv(RESULTS / "summary.csv", summary)
    (RESULTS / "config.json").write_text(json.dumps({
        "seeds": [SEEDS[0], SEEDS[-1]], "loads": LOADS, "dimensions": DIMENSIONS,
        "eta_default": ETA, "eta_scale_check": [0.25, 0.5, 1.0],
        "correlated_flip_probability": FLIP_PROBABILITY,
        "correlated_expected_pairwise_overlap": (1 - 2 * FLIP_PROBABILITY) ** 2,
        "numpy": np.__version__, "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
    }, indent=2), encoding="utf-8")
    plot(summary)
    print(f"Saved {len(rows)} independent runs and {len(summary)} summary rows")


if __name__ == "__main__":
    main()
