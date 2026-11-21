"""Linear associative memory; learning rules live in loss functions."""

import torch

from .model_base import ModelBase


class ModelMemory(ModelBase):
    def __init__(self, d=64, m=32, device="cpu"):
        super().__init__(d, m)
        self.W = torch.nn.Parameter(
            torch.zeros(m, d, dtype=torch.float32, device=device)
        )

    def forward(self, **kwargs):
        q = kwargs["q"]
        self._validate(q, self.d)
        return q @ self.W.T
