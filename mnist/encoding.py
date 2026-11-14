"""Encode normalized MNIST intensities as binary spike trains."""

import numpy as np


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
    image = _validate(image, steps)
    return rng.random((steps, *image.shape)) < image


def latency_encode(image, steps):
    image = _validate(image, steps)
    times = np.rint((1 - image) * (steps - 1)).astype(int)
    return (np.arange(steps)[:, None, None] == times) & (image > 0)


def phase_encode(image, steps, period):
    image = _validate(image, steps)
    if not isinstance(period, int) or not 1 <= period <= steps:
        raise ValueError("period must be an integer between 1 and steps")
    phases = np.rint((1 - image) * (period - 1)).astype(int)
    return (np.arange(steps)[:, None, None] % period == phases) & (image > 0)
