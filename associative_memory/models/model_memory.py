"""Linear associative memory; learning rules live in loss functions."""

import torch


class ModelMemory(torch.nn.Module):
    def __init__(self, d=64, m=32, device="cpu"):
        super().__init__()
        if not isinstance(d, int) or not isinstance(m, int) or min(d, m) <= 0:
            raise ValueError("d and m must be positive integers")
        self.d, self.m = d, m
        self.W = torch.nn.Parameter(
            torch.zeros(m, d, dtype=torch.float32, device=device)
        )

    def forward(self, **kwargs):
        q = kwargs["q"]
        if not isinstance(q, torch.Tensor):
            raise TypeError("inputs must be torch tensors")
        assert q.dtype == torch.float32, "inputs must be float32"
        if q.ndim not in (1, 2) or q.shape[-1] != self.d:
            raise ValueError("invalid input shape")
        if not q.isfinite().all():
            raise ValueError("inputs must be finite")
        return q @ self.W.T
