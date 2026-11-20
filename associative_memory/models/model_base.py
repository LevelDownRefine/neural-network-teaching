"""Shared interface for online key–value associative memory."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ModelBase(ABC):
    """Store W with shape (m, d); training consumes one pair at a time."""

    def __init__(self, d=64, m=32, eta=0.5):
        if not isinstance(d, int) or not isinstance(m, int) or min(d, m) <= 0:
            raise ValueError("d and m must be positive integers")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.d, self.m, self.eta = d, m, float(eta)
        self.W = np.zeros((m, d), dtype=np.float64)
        self._data = None
        self._results = None

    def feed_data(self, data):
        """Accept {k, v} for a training pair or {q} for retrieval.

        Queries may have shape (d,) or (P, d). No normalization is applied.
        """
        if set(data) not in ({"k", "v"}, {"q"}):
            raise ValueError("provide exactly {k, v} or {q}")
        copied = {
            key: np.array(value, dtype=self.W.dtype, copy=True)
            for key, value in data.items()
        }
        if any(not np.isfinite(value).all() for value in copied.values()):
            raise ValueError("data must be finite")
        if "k" in copied:
            if copied["k"].shape != (self.d,) or copied["v"].shape != (self.m,):
                raise ValueError("training requires one k (d,) and one v (m,)")
        else:
            q = copied["q"]
            if q.ndim not in (1, 2) or q.shape[-1] != self.d:
                raise ValueError("q must have shape (d,) or (P, d)")
        self._data = copied
        self._results = None

    def _training_pair(self):
        if self._data is None or "k" not in self._data:
            raise RuntimeError("feed a training pair before optimization")
        self._results = None
        return self._data["k"], self._data["v"]

    @abstractmethod
    def optimize_parameters(self):
        """Apply one online weight update using the current pair."""

    def test(self):
        """Retrieve without changing weights; zero output maps to +1."""
        if self._data is None or "q" not in self._data:
            raise RuntimeError("feed queries before test")
        z = self._data["q"] @ self.W.T
        self._results = {"z": z, "v_hat": np.where(z >= 0, 1, -1)}

    def current_results(self):
        if self._results is None:
            raise RuntimeError("run test before requesting results")
        return {key: value.copy() for key, value in self._results.items()}

    def reset(self):
        self.W.fill(0)
        self._data = self._results = None

    def state_dict(self):
        return {"W": self.W.copy()}

    def load_state_dict(self, state):
        weights = np.asarray(state["W"], dtype=self.W.dtype)
        if weights.shape != self.W.shape or not np.isfinite(weights).all():
            raise ValueError("weights must be finite and match model dimensions")
        self.W[...] = weights
        self._data = self._results = None

    def save(self, path):
        """Save weights and hyperparameters to an NPZ file at the exact path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez(stream, W=self.W, eta=self.eta)

    def load(self, path):
        with np.load(path, allow_pickle=False) as state:
            eta = float(state["eta"])
            if not np.isfinite(eta) or eta <= 0:
                raise ValueError("invalid saved learning rate")
            self.load_state_dict(state)
            self.eta = eta
