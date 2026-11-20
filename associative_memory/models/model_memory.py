"""Delta, Hebbian and no-write associative memory models."""

import numpy as np

from .model_base import ModelBase


class ModelDelta(ModelBase):
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

    def optimize_parameters(self):
        k, v = self._training_pair()
        error = v - self.W @ k
        self.W += self.eta * np.outer(error, k)

    def test(self):
        """Retrieve without changing weights; zero output maps to +1."""
        if self._data is None or "q" not in self._data:
            raise RuntimeError("feed queries before test")
        z = self._data["q"] @ self.W.T
        self._results = {"z": z, "v_hat": np.where(z >= 0, 1, -1)}

    def current_results(self):
        if self._results is None:
            raise RuntimeError("run test before requesting results")
        return self._results

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


class ModelHebbian(ModelDelta):
    def optimize_parameters(self):
        k, v = self._training_pair()
        self.W += self.eta * np.outer(v, k)


class ModelZero(ModelDelta):
    """No-write baseline, initialized at zero like the other models."""

    def optimize_parameters(self):
        self._training_pair()
