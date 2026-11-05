"""Estimate dense convolution and linear floating-point operations per input."""

import argparse
import json
from math import prod
from pathlib import Path

import torch
import yaml
from models.m5 import ModelM5
from models.tssl_cnn import TSSLCNN
from torch import nn

ROOT = Path(__file__).resolve().parent.parent


def count_dense_layers(model, input_size=28):
    """Count dense MACs using the shapes from a CPU forward pass."""
    layers = []
    handles = []

    def record(name):
        def hook(module, inputs, output):
            macs = output.numel() * prod(module.weight.shape[1:])
            layers.append({"name": name, "macs": macs})

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(record(name)))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 1, input_size, input_size))
    finally:
        for handle in handles:
            handle.remove()
    return {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "steps": 1,
        "layers": layers,
        "macs": sum(layer["macs"] for layer in layers),
    }


def count_tssl(config, input_size=28):
    """Count dense weights in the TSSL YAML, including all time steps."""
    steps = config["Network"]["n_steps"]
    channels, height, width = 1, input_size, input_size
    parameters = 0
    layers = []
    for name, layer in config["Layers"].items():
        kind = layer["type"]
        if kind in ("conv", "pooling"):
            kernel = layer["kernel_size"]
            padding = layer.get("padding", 0)
            stride = layer.get("stride", 1 if kind == "conv" else kernel)
            dilation = layer.get("dilation", 1)
            height = (height + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            width = (width + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        if kind == "conv":
            if layer["in_channels"] != channels:
                raise ValueError(f"invalid input channels: {name}")
            weight_count = layer["out_channels"] * channels * kernel * kernel
            channels = layer["out_channels"]
            parameters += weight_count
            layers.append({"name": name, "macs": height * width * weight_count * steps})
        elif kind == "linear":
            if layer["n_inputs"] != channels * height * width:
                raise ValueError(f"invalid input features: {name}")
            weight_count = layer["n_inputs"] * layer["n_outputs"]
            parameters += weight_count
            layers.append({"name": name, "macs": weight_count * steps})
            channels, height, width = layer["n_outputs"], 1, 1
        elif kind not in ("conv", "pooling"):
            raise ValueError(f"unsupported TSSL layer type: {kind}")
    return {
        "parameters": parameters,
        "steps": steps,
        "layers": layers,
        "macs": sum(layer["macs"] for layer in layers),
    }


def all_counts():
    config = yaml.safe_load(
        (ROOT / "models" / "tssl" / "Networks" / "MNIST_CNN.yaml").read_text(
            encoding="utf-8"
        )
    )
    return {
        "logistic": count_dense_layers(nn.Sequential(nn.Flatten(), nn.Linear(784, 10))),
        "m5": count_dense_layers(ModelM5()),
        "tssl_cnn": count_dense_layers(TSSLCNN()),
        "tssl": count_tssl(config),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", choices=("logistic", "m5", "tssl_cnn", "tssl")
    )
    args = parser.parse_args()
    counts = all_counts()
    if args.model:
        counts = {args.model: counts[args.model]}
    output = {
        name: {
            "parameters": record["parameters"],
            "steps": record["steps"],
            "layers": [
                {"name": layer["name"], "flops_approx_per_input": 2 * layer["macs"]}
                for layer in record["layers"]
            ],
            "flops_approx_per_input": 2 * record["macs"],
        }
        for name, record in counts.items()
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
