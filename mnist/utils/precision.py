"""Precision policy shared by inference benchmarks and checkpoint evaluation."""

from contextlib import nullcontext

import torch


def inference_precision(name, device):
    if name == "fp32":
        return nullcontext()
    if name != "bf16":
        raise ValueError(f"unsupported precision: {name}")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 autocast is not supported by the selected CUDA device")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
