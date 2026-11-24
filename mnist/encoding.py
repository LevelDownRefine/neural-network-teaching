"""Encode batches of MNIST images as time-varying SNN inputs."""

import math

import torch


def _rank_order(images, steps):
    flat = images.flatten(start_dim=1)
    sorted_values, order = flat.sort(dim=1, descending=True, stable=True)
    indices = torch.arange(flat.shape[1], device=images.device).expand_as(order)
    starts = torch.cat(
        (
            torch.ones_like(sorted_values[:, :1], dtype=torch.bool),
            sorted_values[:, 1:] != sorted_values[:, :-1],
        ),
        dim=1,
    )
    sorted_ranks = torch.where(starts, indices, 0).cummax(dim=1).values
    ranks = torch.zeros_like(order).scatter(1, order, sorted_ranks)
    active = (flat > 0).sum(dim=1, keepdim=True).clamp_min(1)
    times = torch.div(ranks * steps, active, rounding_mode="floor").clamp_max(steps - 1)
    times = times.reshape(images.shape)
    return (torch.arange(steps, device=images.device) == times.unsqueeze(-1)) & (
        images > 0
    ).unsqueeze(-1)


def _population(images, steps, generator, population_size, population_width):
    if (
        not isinstance(population_size, int)
        or isinstance(population_size, bool)
        or population_size < 2
    ):
        raise ValueError("population_size must be an integer of at least 2")
    if (
        not isinstance(population_width, (int, float))
        or isinstance(population_width, bool)
        or not math.isfinite(population_width)
        or population_width <= 0
    ):
        raise ValueError("population_width must be positive and finite")
    centers = torch.linspace(
        0, 1, population_size, device=images.device, dtype=images.dtype
    )
    values = images.unsqueeze(2)
    response = torch.exp(
        -0.5 * ((values - centers.view(1, 1, -1, 1, 1)) / population_width) ** 2
    )
    response = response * (values > 0)
    samples = torch.rand(
        (*response.shape, steps),
        dtype=images.dtype,
        device=images.device,
        generator=generator,
    )
    spikes = samples < response.unsqueeze(-1)
    batch, channels, height, width = images.shape
    return spikes.reshape(batch, channels * population_size, height, width, steps)


def encode_input(
    images,
    steps,
    method="repeat",
    *,
    generator=None,
    period=None,
    population_size=3,
    population_width=0.25,
):
    """Return [batch, channel, height, width, time] on the input device.

    ``repeat`` preserves continuous values. Spike encodings require
    floating-point intensities in [0, 1] and return zeros and ones in the input
    dtype. Rate and population accept a same-device generator. Population expands
    the channel dimension by population_size; phase requires a period. Spike encoding is
    nondifferentiable with respect to the image; downstream weights can still train.
    """
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channel, height, width]")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if method == "repeat":
        return images.unsqueeze(-1).repeat(1, 1, 1, 1, steps)
    if method not in ("rate", "latency", "phase", "rank_order", "population"):
        raise ValueError(f"unknown encoding method: {method}")
    if not images.is_floating_point():
        raise ValueError("spike encoding requires floating-point intensities")
    if not torch.all(torch.isfinite(images) & (images >= 0) & (images <= 1)):
        raise ValueError("spike encoding requires finite intensities in [0, 1]")

    if method == "rank_order":
        spikes = _rank_order(images, steps)
    elif method == "population":
        spikes = _population(
            images, steps, generator, population_size, population_width
        )
    elif method == "rate":
        samples = torch.rand(
            (*images.shape, steps),
            dtype=images.dtype,
            device=images.device,
            generator=generator,
        )
        spikes = samples < images.unsqueeze(-1)
    else:
        if method == "phase":
            if not isinstance(period, int) or isinstance(period, bool) or period < 1:
                raise ValueError("period must be a positive integer")
            window = period
        else:
            window = steps
        times = ((1 - images) * (window - 1)).round().to(torch.long)
        indices = torch.arange(steps, device=images.device) % window
        spikes = (indices == times.unsqueeze(-1)) & (images > 0).unsqueeze(-1)
    return spikes.to(images.dtype)
