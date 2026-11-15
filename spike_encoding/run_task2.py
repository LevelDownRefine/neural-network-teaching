"""Run the Task 2 experiment grid sequentially and summarize completed runs."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "mnist" / "results"
METHODS = ("rate", "latency", "phase")
STEPS = (10, 20, 50, 100)


def experiment_path(method, steps, seed):
    return RESULTS / f"snntorch_task2_{method}_{steps}_seed{seed}"


def matching_result(path, args, method, steps):
    result = json.loads(path.read_text(encoding="utf-8"))
    config = result["config"]
    expected = {
        "encoding": method,
        "steps": steps,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augment": True,
        "ema": True,
        "max_batches": 0,
        "phase_period": args.phase_period,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing result has different configuration: {path}")
    if result["test"]["count"] != 10_000:
        raise ValueError(f"Existing result is not a full test-set evaluation: {path}")
    return result


def write_report(rows, folder):
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, xlabel, ylabel, xkey, ykeys in (
        ("accuracy_vs_steps", "Time steps", "Test accuracy", "steps", ("accuracy",)),
        (
            "spikes_vs_steps",
            "Time steps",
            "Spikes per image",
            "steps",
            ("input_spikes_per_image", "hidden_spikes_per_image"),
        ),
        (
            "accuracy_vs_spikes",
            "Input spikes per image",
            "Test accuracy",
            "input_spikes_per_image",
            ("accuracy",),
        ),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHODS:
            data = [row for row in rows if row["encoding"] == method]
            for key in ykeys:
                label = method if len(ykeys) == 1 else f"{method} / {key.split('_')[0]}"
                ax.plot(
                    [row[xkey] for row in data],
                    [row[key] for row in data],
                    marker="o",
                    label=label,
                )
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-period", type=int, default=20)
    parser.add_argument(
        "--wait-first",
        action="store_true",
        help="wait for an already-running first experiment",
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.phase_period) < 1:
        parser.error("epochs, batch size, and phase period must be positive")
    folder = RESULTS / f"snntorch_task2_grid_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        for steps in STEPS:
            path = experiment_path(method, steps, args.seed)
            metrics = path / "metrics.json"
            if args.wait_first and (method, steps) == (METHODS[0], STEPS[0]):
                print(
                    f"Waiting for the existing {method}/{steps} experiment", flush=True
                )
                while not metrics.exists():
                    latest = path / "latest.pt"
                    if latest.exists() and time.time() - latest.stat().st_mtime > 3600:
                        raise RuntimeError(f"No checkpoint update for an hour: {path}")
                    time.sleep(30)
            if not metrics.exists():
                tag = f"task2_{method}_{steps}_seed{args.seed}"
                cmd = [
                    sys.executable,
                    "-u",
                    str(ROOT / "mnist" / "train_tssl.py"),
                    "--encoding",
                    method,
                    "--steps",
                    str(steps),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--phase-period",
                    str(args.phase_period),
                    "--augment",
                    "--ema",
                    "--tag",
                    tag,
                ]
                if (path / "latest.pt").exists():
                    cmd.append("--resume")
                print(
                    f"Starting {method}/{steps}; log: {folder / (tag + '.log')}",
                    flush=True,
                )
                with (folder / (tag + ".log")).open("a", encoding="utf-8") as log:
                    subprocess.run(
                        cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
            result = matching_result(metrics, args, method, steps)
            rows.append(
                {
                    "encoding": method,
                    "steps": steps,
                    "accuracy": result["test"]["accuracy"],
                    "input_spikes_per_image": result["spike_activity"]["input"][
                        "spikes_per_image"
                    ],
                    "hidden_spikes_per_image": result["spike_activity"]["hidden"][
                        "spikes_per_image"
                    ],
                    "model_only_batch_ms": result["latency"]["model_only"][
                        "mean_batch_ms"
                    ],
                    "encoding_and_model_batch_ms": result["latency"][
                        "encoding_and_model"
                    ]["mean_batch_ms"],
                    "latency_batch_size": result["latency"]["model_only"]["batch_size"],
                }
            )
            write_report(rows, folder)
            print(
                f"Completed {method}/{steps}; {len(rows)}/{len(METHODS) * len(STEPS)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
