"""Reproducible, independent random streams for all homework experiments."""

import hashlib
import json

import numpy as np


def make_rng(seed, experiment, purpose, *coordinates):
    """Recreate a stream from its identity, independent of call order.

    Example: make_rng(42, 'experiment1', 'data', 64, 32).
    Use separate purposes for data, order, masks, buffer and replay.
    Coordinates are nonnegative integers identifying an experimental condition.
    Create once per task and keep the generator while sampling that task.
    """
    if any(
        not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) or x < 0
        for x in (seed, *coordinates)
    ):
        raise ValueError("seed and coordinates must be nonnegative integers")
    if (
        not isinstance(experiment, str)
        or not experiment
        or not isinstance(purpose, str)
        or not purpose
    ):
        raise ValueError("experiment and purpose must be nonempty strings")
    identity = json.dumps([experiment, purpose], ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    namespace = np.frombuffer(digest, dtype="<u4").tolist()
    sequence = np.random.SeedSequence([int(seed), *namespace, *map(int, coordinates)])
    return np.random.Generator(np.random.PCG64(sequence))
