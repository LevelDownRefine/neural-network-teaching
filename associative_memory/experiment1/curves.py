"""Plot the three experiment metrics with mean and sample SD."""

import numpy as np


def plot_curves(rows, path, subtitle):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for row, d in enumerate(sorted({r["d"] for r in rows})):
        for col, metric in enumerate(("bit_accuracy", "mse", "weight_frobenius")):
            ax = axes[row, col]
            for method in ("Hebbian", "Delta", "Zero"):
                group = [r for r in rows if r["d"] == d and r["method"] == method]
                x = [r["load" if col == 0 else "P"] for r in group]
                y = np.array([r[metric + "_mean"] for r in group])
                sd = np.array([r[metric + "_sd"] for r in group])
                (line,) = ax.plot(x, y, marker="o", label=method)
                ax.fill_between(
                    x,
                    np.maximum(0, y - sd),
                    np.minimum(1, y + sd) if col == 0 else y + sd,
                    color=line.get_color(),
                    alpha=0.1,
                )
            ax.set(title=f"d={d}: {metric}", xlabel="P/d" if col == 0 else "P")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("FP32; 200 trials; mean +/- sample SD")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
