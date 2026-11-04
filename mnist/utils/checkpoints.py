"""Protect experiment outputs and save recoverable checkpoints."""

from pathlib import Path

import torch


def prepare_output(path, resume, allow_resume=True):
    path = Path(path)
    if not resume and any(
        (path / name).exists() for name in ("latest.pt", "best.pt", "metrics.json")
    ):
        hint = "use --resume or a new --tag" if allow_resume else "use a new --tag"
        raise FileExistsError(f"{path} already contains an experiment; {hint}")
    path.mkdir(parents=True, exist_ok=True)


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def cpu_snapshot(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def save_training(state, path):
    path = Path(path)
    atomic_save(state, path / "latest.pt")
    atomic_save(state["best_checkpoint"], path / "best.pt")


def restore_best(state, path):
    path = Path(path)
    best = state.get("best_checkpoint")
    if best is None:
        best = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    atomic_save(best, path / "best.pt")
    return best
