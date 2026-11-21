"""Delta, Hebbian and no-write associative memory models."""

import math

import torch

from .model_base import ModelBase


class ModelDelta(ModelBase):
    def __init__(self, d=64, m=32, eta=0.5, device="cpu"):
        super().__init__(device)
        if not isinstance(d, int) or not isinstance(m, int) or min(d, m) <= 0:
            raise ValueError("d and m must be positive integers")
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.d, self.m, self.eta = d, m, float(eta)
        self.register_buffer(
            "W", torch.zeros(m, d, dtype=torch.float32, device=self.device)
        )
        self._data = None
        self._results = None

    def feed_data(self, data):
        """Accept {k, v} for a training pair or {q} for retrieval.

        Queries may have shape (d,) or (P, d). No normalization is applied.
        """
        if set(data) not in ({"k", "v"}, {"q"}):
            raise ValueError("provide exactly {k, v} or {q}")
        if any(not isinstance(value, torch.Tensor) for value in data.values()):
            raise TypeError("model inputs must be torch tensors")
        inputs = {key: value.to(self.device) for key, value in data.items()}
        assert all(value.dtype == torch.float32 for value in inputs.values()), (
            "inputs must be float32"
        )
        if any(not torch.isfinite(value).all() for value in inputs.values()):
            raise ValueError("data must be finite")
        if "k" in inputs:
            if inputs["k"].shape != (self.d,) or inputs["v"].shape != (self.m,):
                raise ValueError("training requires one k (d,) and one v (m,)")
        else:
            q = inputs["q"]
            if q.ndim not in (1, 2) or q.shape[-1] != self.d:
                raise ValueError("q must have shape (d,) or (P, d)")
        self._data = inputs
        self._results = None

    def _training_pair(self):
        if self._data is None or "k" not in self._data:
            raise RuntimeError("feed a training pair before optimization")
        self._results = None
        return self._data["k"], self._data["v"]

    @torch.no_grad()
    def optimize_parameters(self):
        k, v = self._training_pair()
        error = v - self(k)
        self.W += self.eta * torch.outer(error, k)

    def forward(self, q):
        return q @ self.W.T

    @torch.no_grad()
    def test(self):
        """Retrieve without changing weights; zero output maps to +1."""
        if self._data is None or "q" not in self._data:
            raise RuntimeError("feed queries before test")
        self.eval()
        try:
            z = self(self._data["q"])
        finally:
            self.train()
        self._results = {"z": z, "v_hat": torch.where(z >= 0, 1, -1)}

    def current_results(self):
        if self._results is None:
            raise RuntimeError("run test before requesting results")
        return {key: value.detach().cpu() for key, value in self._results.items()}

    def reset(self):
        self.W.zero_()
        self._data = self._results = None

    def load_state_dict(self, state):
        if set(state) != {"W"}:
            raise ValueError("state must contain exactly W")
        weights = state["W"]
        assert weights.dtype == torch.float32, "weights must be float32"
        if weights.shape != self.W.shape or not weights.isfinite().all():
            raise ValueError("weights must be finite and match model dimensions")
        result = super().load_state_dict(state)
        self._data = self._results = None
        return result


class ModelHebbian(ModelDelta):
    @torch.no_grad()
    def optimize_parameters(self):
        k, v = self._training_pair()
        self.W += self.eta * torch.outer(v, k)


class ModelZero(ModelDelta):
    """No-write baseline, initialized at zero like the other models."""

    @torch.no_grad()
    def optimize_parameters(self):
        self._training_pair()
