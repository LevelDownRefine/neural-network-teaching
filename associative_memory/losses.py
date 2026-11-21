"""Objectives whose SGD steps reproduce the homework plasticity rules."""

import torch


def _validate(pred, target):
    for tensor in (pred, target):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("loss inputs must be torch tensors")
        assert tensor.dtype == torch.float32, "loss inputs must be float32"
        if tensor.ndim != 1 or not tensor.isfinite().all():
            raise ValueError("loss requires finite vectors for one training pair")
    if pred.shape != target.shape:
        raise ValueError("prediction and target shapes must match")


def delta_loss(pred, target):
    """Half squared error, summed over output coordinates."""
    _validate(pred, target)
    return 0.5 * (pred - target).square().sum()


def hebbian_loss(pred, target):
    """Negative correlation objective; unbounded, not an error metric."""
    _validate(pred, target)
    return -(pred * target).sum()
