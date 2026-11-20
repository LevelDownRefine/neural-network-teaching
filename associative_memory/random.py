"""Shared random generator factory for homework experiments."""

import numpy as np


def make_rng(seed):
    """Create a reproducible NumPy generator from an explicit seed."""
    return np.random.default_rng(seed)
