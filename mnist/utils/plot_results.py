"""Plot confusion matrices from saved full-test evaluations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utils.results import discover

ROOT = Path(__file__).resolve().parent.parent
for record in discover(ROOT / "results"):
    model = record["name"]
    path = record["path"]
    metrics = record["metrics"]
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1200)
    for y in range(10):
        for x in range(10):
            ax.text(
                x,
                y,
                str(matrix[y, x]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[y, x] > 600 else "black",
            )
    ax.set(
        xticks=range(10),
        yticks=range(10),
        xlabel="Predicted digit",
        ylabel="True digit",
        title=f"{model}: test accuracy {metrics['accuracy']:.2%}",
    )
    fig.colorbar(im, ax=ax, label="Number of images")
    fig.savefig(path.parent / "confusion_matrix.png", dpi=150)
    plt.close(fig)
