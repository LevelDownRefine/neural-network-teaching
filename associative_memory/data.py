"""Random key-value pairs and initial presentation order."""

import numpy as np


def generate_pairs(rng, p, d, m):
    """Return K (p,d), V (p,m), and a permutation of p indices."""
    keys = rng.choice([-1.0, 1.0], (p, d)) / np.sqrt(d)
    values = rng.choice([-1.0, 1.0], (p, m))
    order = rng.permutation(p)
    return keys, values, order
