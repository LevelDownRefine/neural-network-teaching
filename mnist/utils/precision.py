"""Shared inference precision and execution policies."""

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


def inference_engine(predict, engine):
    if engine == "eager":
        return predict
    if engine == "inductor":
        return torch.compile(predict, backend="inductor", fullgraph=True)
    raise ValueError(f"unsupported engine: {engine}")
