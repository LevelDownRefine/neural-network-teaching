"""Shared inference execution policy."""

import torch


def inference_engine(predict, engine):
    if engine == "eager":
        return predict
    if engine == "inductor":
        return torch.compile(predict, backend="inductor", fullgraph=True)
    raise ValueError(f"unsupported engine: {engine}")
