"""Encode batches of MNIST images as time-varying SNN inputs."""

import torch


def encode_input(images, steps, method="repeat", *, generator=None, period=None):
    """Return [batch, channel, height, width, time] on the input device.

    ``repeat`` preserves continuous values. The three spike encodings require
    floating-point intensities in [0, 1] and return zeros and ones in the input
    dtype. For reproducible Rate samples, provide a generator on the same device.
    Phase coding requires an explicit period in time steps. Spike encoding is
    nondifferentiable with respect to the image; downstream weights can still train.
    """
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channel, height, width]")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if method == "repeat":
        return images.unsqueeze(-1).repeat(1, 1, 1, 1, steps)
    if method not in ("rate", "latency", "phase"):
        raise ValueError(f"unknown encoding method: {method}")
    if not images.is_floating_point():
        raise ValueError("spike encoding requires floating-point intensities")
    if not torch.all(torch.isfinite(images) & (images >= 0) & (images <= 1)):
        raise ValueError("spike encoding requires finite intensities in [0, 1]")

    if method == "rate":
        samples = torch.rand(
            (*images.shape, steps),
            dtype=images.dtype,
            device=images.device,
            generator=generator,
        )
        spikes = samples < images.unsqueeze(-1)
    else:
        if method == "phase":
            if (
                not isinstance(period, int)
                or isinstance(period, bool)
                or not 1 <= period <= steps
            ):
                raise ValueError("period must be an integer between 1 and steps")
            window = period
        else:
            window = steps
        times = ((1 - images) * (window - 1)).round().to(torch.long)
        indices = torch.arange(steps, device=images.device) % window
        spikes = (indices == times.unsqueeze(-1)) & (images > 0).unsqueeze(-1)
    return spikes.to(images.dtype)
