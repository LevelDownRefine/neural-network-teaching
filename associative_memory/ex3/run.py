"""HW3 Experiment 3: compare sequential Delta learning with old-memory replay."""

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import torch

from associative_memory.data import generate_pairs
from associative_memory.models import ModelMemory
from associative_memory.train import DeltaTrainer, train_epoch

D, M, PAIRS = 64, 32, 32
A_EPOCHS, B_EPOCHS = 4, 4
LEARNING_RATE = 0.5
CONDITIONS = ("sequential", "matched", "replay_4", "replay_16", "replay_32")
CAPACITIES = {
    "sequential": 0,
    "matched": 0,
    "replay_4": 4,
    "replay_16": 16,
    "replay_32": 32,
}
LABELS = ("Sequential", "Update matched", "Replay b=4", "Replay b=16", "Replay b=32")
SCOPES = ("a_accuracy", "b_accuracy", "a_in_pool_accuracy", "a_outside_pool_accuracy")


@torch.no_grad()
def correct_bits(model, keys, values):
    model.eval()
    prediction = torch.where(model(q=keys) >= 0, 1, -1)
    return prediction.eq(values)


def score_checkpoint(model, keys_a, values_a, keys_b, values_b, pool_mask):
    correct_a = correct_bits(model, keys_a, values_a)
    correct_b = correct_bits(model, keys_b, values_b)
    return {
        "a_accuracy": correct_a.float().mean().item(),
        "b_accuracy": correct_b.float().mean().item(),
        "a_in_pool_accuracy": (
            correct_a[pool_mask].float().mean().item() if pool_mask.any() else None
        ),
        "a_outside_pool_accuracy": (
            correct_a[~pool_mask].float().mean().item()
            if pool_mask.any() and (~pool_mask).any()
            else None
        ),
    }


def run_trial(trial, seed):
    data_seed, *replay_seeds = seed.spawn(1 + 3)
    rng = np.random.default_rng(data_seed)
    ka, va, first_a_order = generate_pairs(rng, PAIRS, D, M)
    kb, vb, first_b_order = generate_pairs(rng, PAIRS, D, M)
    keys_a, values_a, keys_b, values_b = [
        torch.as_tensor(array, dtype=torch.float32) for array in (ka, va, kb, vb)
    ]
    a_orders = [first_a_order] + [rng.permutation(PAIRS) for _ in range(A_EPOCHS - 1)]
    b_orders = [first_b_order] + [rng.permutation(PAIRS) for _ in range(B_EPOCHS - 1)]

    model = ModelMemory(d=D, m=M)
    trainer = DeltaTrainer(model, lr=LEARNING_RATE)
    for order in a_orders:
        train_epoch(trainer, keys_a, values_a, order)
    snapshot = deepcopy(model.state_dict())

    rows, costs = [], []
    replay_streams = iter(replay_seeds)
    for condition in CONDITIONS:
        capacity = CAPACITIES[condition]
        model.load_state_dict(snapshot)
        trainer = DeltaTrainer(model, lr=LEARNING_RATE)
        pool_mask = torch.zeros(PAIRS, dtype=torch.bool)
        pool_keys, pool_values = keys_a[:0], values_a[:0]
        if capacity:
            replay_rng = np.random.default_rng(next(replay_streams))
            pool_indices = replay_rng.choice(PAIRS, size=capacity, replace=False)
            pool_mask[pool_indices] = True
            # Advanced indexing materializes a separate buffer of k/v pairs.
            pool_keys, pool_values = keys_a[pool_indices], values_a[pool_indices]
            draws = replay_rng.integers(capacity, size=(B_EPOCHS, PAIRS))

        b_updates = 0
        for b_epoch in range(B_EPOCHS + 1):
            rows.append(
                {
                    "trial": trial,
                    "condition": condition,
                    "capacity": capacity,
                    "b_epoch": b_epoch,
                    "b_updates": b_updates,
                    **score_checkpoint(
                        model, keys_a, values_a, keys_b, values_b, pool_mask
                    ),
                }
            )
            if b_epoch == B_EPOCHS:
                break
            for position, index in enumerate(b_orders[b_epoch]):
                trainer.step({"q": keys_b[index]}, values_b[index])
                b_updates += 1
                if condition == "matched":
                    trainer.step({"q": keys_b[index]}, values_b[index])
                    b_updates += 1
                elif capacity:
                    replay_index = int(draws[b_epoch, position])
                    trainer.step(
                        {"q": pool_keys[replay_index]}, pool_values[replay_index]
                    )
                    b_updates += 1

        costs.append(
            {
                "condition": condition,
                "capacity": capacity,
                "a_updates": A_EPOCHS * PAIRS,
                "b_updates": b_updates,
                "total_updates": A_EPOCHS * PAIRS + b_updates,
                "weight_scalars": model.W.numel(),
                "weight_bytes": model.W.nbytes,
                "buffer_scalars": pool_keys.numel() + pool_values.numel(),
                "buffer_bytes": pool_keys.nbytes + pool_values.nbytes,
                "total_bytes": model.W.nbytes + pool_keys.nbytes + pool_values.nbytes,
            }
        )
    return rows, costs


def summarize(rows):
    summary = []
    for condition in CONDITIONS:
        for epoch in range(B_EPOCHS + 1):
            group = [
                r for r in rows if r["condition"] == condition and r["b_epoch"] == epoch
            ]
            row = {
                key: group[0][key]
                for key in ("condition", "capacity", "b_epoch", "b_updates")
            }
            row["trials"] = len(group)
            for scope in SCOPES:
                values = [r[scope] for r in group if r[scope] is not None]
                row[scope + "_mean"] = float(np.mean(values)) if values else None
                row[scope + "_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
            summary.append(row)
    return summary


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, output, trials):
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(14.7, 5.2))
        fig.subplots_adjust(top=0.78, bottom=0.19, left=0.06, right=0.983, wspace=0.29)
        colors = ("#586474", "#283d57", "#df7e24", "#11877d", "#4365bd")
        for ax, scope, title in zip(
            axes[:2], SCOPES[:2], ("A   Old memories", "B   New memories")
        ):
            for condition, label, color in zip(CONDITIONS, LABELS, colors):
                series = [r for r in summary if r["condition"] == condition]
                means = np.array([r[scope + "_mean"] for r in series]) * 100
                spread = np.array([r[scope + "_sd"] or 0 for r in series]) * 100
                epochs = [r["b_epoch"] for r in series]
                ax.plot(
                    epochs,
                    means,
                    marker="o",
                    color=color,
                    label=label,
                    linewidth=1.9,
                    markersize=4,
                    linestyle="--" if condition == "matched" else "-",
                )
                ax.fill_between(
                    epochs, means - spread, means + spread, color=color, alpha=0.09
                )
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set(xlabel="B passes completed (0 = after A)", ylim=(45, 103))
            ax.set_xticks(range(B_EPOCHS + 1))
        axes[0].set_ylabel("Bit accuracy (%)")

        ax = axes[2]
        bars = [r for r in summary if r["b_epoch"] == B_EPOCHS and r["capacity"]]
        for i, r in enumerate(bars):
            for j, (scope, label, color) in enumerate(
                (
                    ("a_in_pool_accuracy", "In pool", "#297d8d"),
                    ("a_outside_pool_accuracy", "Outside pool", "#e4a44f"),
                )
            ):
                mean, sd = r[scope + "_mean"], r[scope + "_sd"]
                if mean is None:
                    ax.text(
                        i + 0.17, 55, "N/A", ha="center", color="#687584", fontsize=9
                    )
                else:
                    ax.errorbar(
                        i + (-0.17 if j == 0 else 0.17),
                        mean * 100,
                        yerr=(sd or 0) * 100,
                        fmt="o",
                        markersize=7,
                        capsize=4,
                        color=color,
                        label=label if i == 0 else None,
                    )
        ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=9)
        ax.set_title("C   Old memories after B", loc="left", fontweight="bold")
        ax.set(
            xlabel="Replay pool size",
            xlim=(-0.45, 2.45),
            ylim=(45, 103),
            ylabel="Bit accuracy (%)",
            xticks=range(3),
            xticklabels=("4", "16", "32"),
        )
        ax.grid(False, axis="x")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.88),
            ncol=5,
            frameon=False,
        )
        fig.suptitle(
            "Sequential learning and old-memory replay",
            fontsize=20,
            fontweight="bold",
            y=0.975,
        )
        fig.text(
            0.06,
            0.05,
            f"{trials} independent trials. Mean ± 1 sample SD. Identical A snapshot and B order within each trial.",
            color="#45515e",
            fontsize=10,
        )
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results")
    )
    args = parser.parse_args()
    if args.trials < 1 or args.seed < 0:
        parser.error("--trials must be positive and --seed must be nonnegative")
    torch.set_num_threads(1)
    rows = []
    costs = None
    for trial, seed in enumerate(
        np.random.SeedSequence(args.seed).spawn(args.trials), 1
    ):
        trial_rows, trial_costs = run_trial(trial, seed)
        if costs is not None and trial_costs != costs:
            raise RuntimeError("Update or storage totals changed between trials")
        rows.extend(trial_rows)
        costs = trial_costs
        if trial % 25 == 0 or trial == args.trials:
            print(f"Completed {trial}/{args.trials} trials", flush=True)

    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "accuracy_by_trial.csv", rows)
    save_csv(args.output / "accuracy_summary.csv", summary)
    save_csv(args.output / "update_and_storage_cost.csv", costs)
    plot_results(summary, args.output / "replay_results.png", args.trials)
    for r in summary:
        if r["b_epoch"] in (0, B_EPOCHS):
            print(
                f"{r['condition']:>10} | B epoch {r['b_epoch']} | "
                f"A={r['a_accuracy_mean']:.4%} | B={r['b_accuracy_mean']:.4%}"
            )
    print("Saved to", args.output.resolve())


if __name__ == "__main__":
    main()
