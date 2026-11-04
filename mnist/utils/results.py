"""Read completed experiments and older saved evaluations consistently."""

import json
from pathlib import Path


def discover(results_dir):
    results_dir = Path(results_dir)
    for folder in sorted(results_dir.iterdir()) if results_dir.exists() else ():
        if not folder.is_dir() or folder.name == "comparison":
            continue
        path = folder / "metrics.json"
        if not path.exists() and folder.name == "tssl":
            path = folder / "classification.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if config.get("max_batches") or (not config and "smoke" in folder.name):
            continue
        metrics = payload.get("test", payload)
        if "confusion_matrix" not in metrics:
            continue
        metrics = dict(metrics)
        auc_path = folder / "auc.json"
        if metrics.get("macro_auc_ovr") is None and auc_path.exists():
            extra = json.loads(auc_path.read_text(encoding="utf-8"))
            if len(extra["per_class_auc"]) != len(metrics["per_class"]):
                raise ValueError(f"per-class AUC count mismatch: {folder.name}")
            metrics["macro_auc_ovr"] = extra["macro_auc_ovr"]
            metrics["per_class"] = [
                dict(row, auc=auc)
                for row, auc in zip(metrics["per_class"], extra["per_class_auc"])
            ]
        selection = payload.get("selection")
        if selection is None:
            if config.get("official_protocol"):
                selection = "official test accuracy (test set used for selection)"
            else:
                selection = "validation accuracy" if config else "unknown (legacy)"
        yield {
            "name": folder.name,
            "path": path,
            "metrics": metrics,
            "config": config,
            "model": payload.get("model", payload.get("experiment", folder.name)),
            "seed": config.get("seed"),
            "split": payload.get("split"),
            "selection": selection,
        }
