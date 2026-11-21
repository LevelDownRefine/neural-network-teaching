"""Linear associative memory network used by the model controller."""

import torch
from torch import nn


class MemoryNetwork(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.register_buffer("W", torch.zeros(m, d, dtype=torch.float32))

    def forward(self, q):
        return q @ self.W.T
