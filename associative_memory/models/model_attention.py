"""Softmax attention over explicitly stored key-value pairs."""

import math

import torch
from torch.nn import functional as F

from .model_base import ModelBase


class ModelAttention(ModelBase):
    """Each update appends a pair; retrieval uses scaled dot-product attention."""

    def __init__(self, d=64, m=32, device="cpu"):
        super().__init__(device)
        if not isinstance(d, int) or not isinstance(m, int) or min(d, m) <= 0:
            raise ValueError("d and m must be positive integers")
        self.d, self.m = d, m
        self.register_buffer("K", torch.empty(0, d, device=self.device))
        self.register_buffer("V", torch.empty(0, m, device=self.device))
        self._data = None
        self._results = None

    def init_train(self):
        self.train()

    def feed_data(self, data):
        if set(data) not in ({"k", "v"}, {"q"}):
            raise ValueError("provide exactly {k, v} or {q}")
        if any(not isinstance(value, torch.Tensor) for value in data.values()):
            raise TypeError("model inputs must be torch tensors")
        inputs = {key: value.to(self.K.device) for key, value in data.items()}
        assert all(value.dtype == torch.float32 for value in inputs.values()), (
            "inputs must be float32"
        )
        if any(not value.isfinite().all() for value in inputs.values()):
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

    @torch.no_grad()
    def optimize_parameters(self):
        if self._data is None or "k" not in self._data:
            raise RuntimeError("feed a pair before writing")
        self.K = torch.cat((self.K, self._data["k"].unsqueeze(0)))
        self.V = torch.cat((self.V, self._data["v"].unsqueeze(0)))
        self._results = None

    @torch.no_grad()
    def test(self):
        if self._data is None or "q" not in self._data:
            raise RuntimeError("feed queries before test")
        if len(self.K) == 0:
            raise RuntimeError("write at least one pair before retrieval")
        q = self._data["q"]
        self.eval()
        try:
            z = F.scaled_dot_product_attention(
                q.reshape(1, -1, self.d),
                self.K.unsqueeze(0),
                self.V.unsqueeze(0),
                dropout_p=0.0,
                scale=1 / math.sqrt(self.d),
            )[0]
            if q.ndim == 1:
                z = z[0]
        finally:
            self.train()
        self._results = {"z": z, "v_hat": torch.where(z >= 0, 1, -1)}

    def current_results(self):
        if self._results is None:
            raise RuntimeError("run test before requesting results")
        return {key: value.detach().cpu() for key, value in self._results.items()}

    def reset(self):
        self.K = self.K.new_empty(0, self.d)
        self.V = self.V.new_empty(0, self.m)
        self._data = self._results = None

    def load_state_dict(self, state):
        if set(state) != {"K", "V"}:
            raise ValueError("state must contain exactly K and V")
        keys, values = state["K"], state["V"]
        assert keys.dtype == values.dtype == torch.float32, "state must be float32"
        if (
            keys.ndim != 2
            or values.ndim != 2
            or keys.shape[1] != self.d
            or values.shape[1] != self.m
            or len(keys) != len(values)
            or not keys.isfinite().all()
            or not values.isfinite().all()
        ):
            raise ValueError("invalid key-value state")
        # Validate and prepare both arrays before replacing the variable-sized buffers.
        keys = keys.detach().to(self.K.device).clone()
        values = values.detach().to(self.V.device).clone()
        self.K, self.V = keys, values
        self._data = self._results = None
