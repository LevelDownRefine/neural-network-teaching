"""Shared dimension and FP32 input validation for PyTorch modules."""

import torch


class ModelBase(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        if not isinstance(d, int) or not isinstance(m, int) or min(d, m) <= 0:
            raise ValueError("d and m must be positive integers")
        self.d, self.m = d, m

    def _validate(self, tensor, width, single=False):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("inputs must be torch tensors")
        assert tensor.dtype == torch.float32, "inputs must be float32"
        if tensor.ndim not in ((1,) if single else (1, 2)) or tensor.shape[-1] != width:
            raise ValueError("invalid input shape")
        if not tensor.isfinite().all():
            raise ValueError("inputs must be finite")
