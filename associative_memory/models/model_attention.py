"""Unnormalized dot-product attention, equivalent to zero-initialized Hebbian memory."""

import math

import torch


class ModelAttention(torch.nn.Module):
    """Caller owns row-major K/V; no softmax, projection or sqrt(d) scaling."""

    def __init__(self, eta=0.5):
        super().__init__()
        if not math.isfinite(eta) or eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        self.eta = float(eta)

    def forward(self, **kwargs):
        q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
        self._validate_inputs(q, k, v)
        return self.eta * ((q @ k.T) @ v)

    @staticmethod
    def _validate_inputs(q, k, v):
        for tensor in (q, k, v):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("inputs must be torch tensors")
            assert tensor.dtype == torch.float32, "inputs must be float32"
            if not tensor.isfinite().all():
                raise ValueError("inputs must be finite")
        if q.ndim not in (1, 2) or k.ndim != 2 or v.ndim != 2:
            raise ValueError("q must be a vector or matrix; k and v must be matrices")
        if q.shape[-1] != k.shape[1] or len(k) != len(v):
            raise ValueError("query/key dimensions and key/value lengths must match")
        if not (q.device == k.device == v.device):
            raise ValueError("inputs must be on the same device")
