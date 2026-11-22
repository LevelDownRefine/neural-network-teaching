"""Reproduce Homework 3 experiment 2 from the repository root.

Run: python -m associative_memory.ex2.run
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
from associative_memory.train import DeltaTrainer, HebbianTrainer, train_epoch

OUTPUT = Path(__file__).resolve().parent / "results"
OCCLUSIONS = (0.0, 0.25, 0.5, 0.75)
METHODS = {"Hebbian": HebbianTrainer, "Delta": DeltaTrainer}
COLORS = {"Hebbian": "#2563a6", "Delta": "#c65a24"}
CONFIG = {
    "d": 64, "m": 32, "pairs": 32, "eta": 0.5, "epochs": 1,
    "masks_per_memory": 10, "main_seed": 42,
    "repeat_seeds": list(range(20)), "occlusions": OCCLUSIONS,
    "example_memory_indices": [0, 1, 2], "example_mask_index": 0,
}


def experiment(seed):
    """One independent dataset; share order and queries between methods."""
    rng = np.random.default_rng(seed)
    keys, values, order = generate_pairs(rng, 32, 64, 32)
    masks = np.zeros((4, 32, 10, 64), dtype=bool)
    for level, p in enumerate(OCCLUSIONS):
        keep = round((1 - p) * 64)
        for memory in range(32):
            for trial in range(10):
                masks[level, memory, trial, rng.choice(64, keep, replace=False)] = True
    queries = masks * keys[None, :, None, :]
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    np.testing.assert_array_equal(masks.sum(axis=-1),
                                  np.broadcast_to(np.array([64, 48, 32, 16])[:, None, None],
                                                  (4, 32, 10)))
    np.testing.assert_allclose(np.linalg.norm(queries, axis=-1), 1)
    weights, outputs, rows = [], [], []
    tensor_keys = torch.from_numpy(keys.astype(np.float32))
    tensor_values = torch.from_numpy(values.astype(np.float32))
    tensor_queries = torch.from_numpy(queries.reshape(-1, 64).astype(np.float32))
    for name, trainer_cls in METHODS.items():
        model = ModelMemory(d=64, m=32)
        trainer = trainer_cls(model, lr=0.5)
        train_epoch(trainer, tensor_keys, tensor_values, order)
        frozen = model.W.detach().numpy().copy()
        model.eval()
        with torch.no_grad():
            output = model(q=tensor_queries).numpy().reshape(4, 32, 10, 32)
        np.testing.assert_array_equal(model.W.detach().numpy(), frozen)
        assert np.isfinite(output).all()
        # An independent closed form checks the Hebbian training implementation.
        if name == "Hebbian":
            np.testing.assert_allclose(frozen, 0.5 * values.T @ keys, atol=2e-6)
        predictions = np.where(output >= 0, 1, -1)
        for level, p in enumerate(OCCLUSIONS):
            accuracy = np.mean(predictions[level] == values[:, None, :])
            mse = np.mean((output[level] - values[:, None, :]) ** 2)
            exact = np.mean(np.all(predictions[level] == values[:, None, :], axis=-1))
            rows.append({"seed": seed, "method": name, "occlusion": p,
                         "bit_accuracy": float(accuracy), "mse": float(mse),
                         "exact_recall_rate": float(exact),
                         "output_power": float(np.mean(output[level] ** 2)),
                         "target_alignment": float(np.mean(output[level] * values[:, None, :]))})
        weights.append(frozen)
        outputs.append(output)
    return rows, dict(keys=keys, values=values, order=order, masks=masks,
                      queries=queries, weights=np.array(weights), z=np.array(outputs))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows, repeats):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for name in METHODS:
        selected = [r for r in rows if r["method"] == name]
        for ax, metric in zip(axes, ("bit_accuracy", "mse")):
            ax.plot(OCCLUSIONS, [r[metric] for r in selected], "o-",
                    color=COLORS[name], label=name)
    axes[0].set(ylabel="Bit accuracy", ylim=(0.45, 1))
    axes[0].axhline(0.5, color="gray", ls=":", label="Chance (expected)")
    axes[1].set(ylabel="MSE of continuous output")
    for ax in axes:
        ax.set(xlabel="Occluded coordinates (%)", xticks=OCCLUSIONS,
               xticklabels=["0", "25", "50", "75"])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(OUTPUT / "main_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for name in METHODS:
        for ax, metric in zip(axes, ("bit_accuracy", "mse")):
            groups = [[r[metric] for r in repeats if r["method"] == name
                       and r["occlusion"] == p] for p in OCCLUSIONS]
            ax.errorbar(OCCLUSIONS, np.mean(groups, axis=1),
                        yerr=np.std(groups, axis=1, ddof=1), fmt="o-", capsize=4,
                        label=name, color=COLORS[name])
    axes[0].set(ylabel="Bit accuracy", ylim=(0.45, 1))
    axes[1].set(ylabel="MSE of continuous output")
    for ax in axes:
        ax.set(xlabel="Occluded coordinates (%)", xticks=OCCLUSIONS,
               xticklabels=["0", "25", "50", "75"])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(OUTPUT / "repeat_curves.png", dpi=180)
    plt.close(fig)


def plot_examples(raw):
    # IDs and mask index are fixed before inspecting retrieval outcomes.
    for memory in CONFIG["example_memory_indices"]:
        fig, axes = plt.subplots(4, 4, figsize=(9, 8), layout="constrained")
        for level, p in enumerate(OCCLUSIONS):
            cue = np.sign(raw["queries"][level, memory, 0]).reshape(8, 8)
            images = [cue, raw["values"][memory].reshape(4, 8)]
            for output in raw["z"]:
                images.append(np.where(output[level, memory, 0] >= 0, 1, -1).reshape(4, 8))
            for column, (ax, pixels) in enumerate(zip(axes[level], images)):
                ax.imshow(pixels, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if column == 0:
                    ax.set_ylabel(f"Occlusion {p:.0%}")
                if column >= 2:
                    correct = np.mean(pixels.ravel() == raw["values"][memory])
                    ax.set_xlabel(f"{correct * 100:.3f}% bits correct")
            if level == 0:
                for ax, title in zip(axes[0], ("Partial cue (8 x 8)", "True value (4 x 8)",
                                              "Hebbian recall", "Delta recall")):
                    ax.set_title(title)
        fig.suptitle(f"Memory {memory}; mask 0; seed 42 | black=-1, gray=masked, white=+1")
        fig.savefig(OUTPUT / f"example_{memory}.png", dpi=160)
        plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, raw = experiment(42)
    repeats = []
    for seed in CONFIG["repeat_seeds"]:
        seed_rows, _ = experiment(seed)
        repeats.extend(seed_rows)
    summary = []
    for name in METHODS:
        for p in OCCLUSIONS:
            selected = [r for r in repeats if r["method"] == name and r["occlusion"] == p]
            row = {"method": name, "occlusion": p}
            for metric in ("bit_accuracy", "mse", "exact_recall_rate"):
                samples = [r[metric] for r in selected]
                row[f"{metric}_mean"] = float(np.mean(samples))
                row[f"{metric}_sd"] = float(np.std(samples, ddof=1))
            summary.append(row)
    write_csv(OUTPUT / "main_metrics.csv", rows)
    write_csv(OUTPUT / "repeat_metrics.csv", repeats)
    write_csv(OUTPUT / "repeat_summary.csv", summary)
    np.savez_compressed(OUTPUT / "main_raw.npz", **raw)
    config = {**CONFIG, "numpy_version": np.__version__, "torch_version": torch.__version__,
              "matplotlib_version": matplotlib.__version__,
              "raw_method_axis": list(METHODS),
              "raw_query_axes": ["occlusion", "memory", "mask", "coordinate"],
              "raw_z_axes": ["method", "occlusion", "memory", "mask", "content_bit"]}
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plot_curves(rows, repeats)
    plot_examples(raw)
    print(json.dumps(rows, indent=2))
    print("20-seed replication summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
