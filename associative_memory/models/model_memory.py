"""Delta, Hebbian and no-write associative memory models."""

import numpy as np
import torch

from .model_base import ModelBase


class ModelDelta(ModelBase):
    def __init__(self, d=64, m=32, eta=0.5):
        super().__init__()
        if not isinstance(d, int) or not isinstance(m, int) or min(d, m) <= 0:
            raise ValueError("d and m must be positive integers")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.d, self.m, self.eta = d, m, float(eta)
        self.register_buffer("W", torch.zeros(m, d, dtype=torch.float64))
        self._data = None
        self._results = None

    def feed_data(self, data):
        """Accept {k, v} for a training pair or {q} for retrieval.

        Queries may have shape (d,) or (P, d). No normalization is applied.
        """
        if set(data) not in ({"k", "v"}, {"q"}):
            raise ValueError("provide exactly {k, v} or {q}")
        copied = {
            key: torch.as_tensor(value, dtype=self.W.dtype, device=self.W.device)
            .detach()
            .clone()
            for key, value in data.items()
        }
        if any(not torch.isfinite(value).all() for value in copied.values()):
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
        self.W += self.eta * torch.outer(error, k)

    def forward(self, q):
        return q @ self.W.T

    def test(self):
        """Retrieve without changing weights; zero output maps to +1."""
        if self._data is None or "q" not in self._data:
            raise RuntimeError("feed queries before test")
        z = self(self._data["q"]).detach().cpu().numpy()
        self._results = {"z": z, "v_hat": np.where(z >= 0, 1, -1)}

    def current_results(self):
        if self._results is None:
            raise RuntimeError("run test before requesting results")
        return self._results

    def reset(self):
        self.W.zero_()
        self._data = self._results = None

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Replay branches need independent snapshots, including in nested modules.
        destination[prefix + "W"] = destination[prefix + "W"].clone()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "W" in state_dict:
            weights = torch.as_tensor(state_dict["W"])
            if weights.shape != self.W.shape or not weights.isfinite().all():
                raise ValueError("weights must be finite and match model dimensions")
            state_dict = dict(state_dict, W=weights.clone())
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._data = self._results = None
        return result


class ModelHebbian(ModelDelta):
    def optimize_parameters(self):
        k, v = self._training_pair()
        self.W += self.eta * torch.outer(v, k)


class ModelZero(ModelDelta):
    """No-write baseline, initialized at zero like the other models."""

    def optimize_parameters(self):
        self._training_pair()
