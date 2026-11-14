"""Prepare static images for SNN input over multiple time steps."""

import numpy as np
import torch


def repeat_static_input(images: torch.Tensor, steps: int) -> torch.Tensor:
    """Repeat a batch of continuous images as [batch, channel, height, width, time]."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channel, height, width]")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return images.unsqueeze(-1).repeat(1, 1, 1, 1, steps)


def _validate(image, steps):
    image = np.asarray(image, dtype=np.float32)
    if (
        image.ndim != 2
        or not np.isfinite(image).all()
        or np.any((image < 0) | (image > 1))
    ):
        raise ValueError("image must be a finite 2D array with intensities in [0, 1]")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return image


def rate_encode(image, steps, rng):
    """Encode one [height, width] image as [time, height, width] spikes."""
    image = _validate(image, steps)
    return rng.random((steps, *image.shape)) < image


def latency_encode(image, steps):
    """Encode one image with at most one spike per nonzero pixel."""
    image = _validate(image, steps)
    times = np.rint((1 - image) * (steps - 1)).astype(int)
    return (np.arange(steps)[:, None, None] == times) & (image > 0)


def phase_encode(image, steps, period):
    """Encode one image as repeating spikes with intensity-dependent phase."""
    image = _validate(image, steps)
    if not isinstance(period, int) or not 1 <= period <= steps:
        raise ValueError("period must be an integer between 1 and steps")
    phases = np.rint((1 - image) * (period - 1)).astype(int)
    return (np.arange(steps)[:, None, None] % period == phases) & (image > 0)
