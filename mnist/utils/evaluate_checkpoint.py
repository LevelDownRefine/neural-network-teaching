"""Evaluate one saved MNIST checkpoint without retraining or selecting on test data."""

import argparse
import json
import sys
from pathlib import Path

import torch
from models.m5 import ModelM5
from models.tssl_cnn import TSSLCNN
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.evaluation import evaluate_batches
from utils.inference import inference_engine
from utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parent.parent


def load_predictor(model_name, weights, device):
    """Load a checkpoint and return a function that predicts on device tensors."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if model_name == "tssl":
        if device.type != "cuda":
            raise RuntimeError("TSSL source requires CUDA")
        import yaml

        src = ROOT / "models" / "tssl"
        sys.path.insert(0, str(src))
        import cnns

        cfg = yaml.safe_load((src / "Networks" / "MNIST_CNN.yaml").read_text())
        hidden = checkpoint["net"]["my_parameters.2"].shape[0]
        cfg["Layers"]["FC_1"]["n_outputs"] = hidden
        cfg["Layers"]["output"]["n_inputs"] = hidden
        nc = cfg["Network"]
        net = cnns.Network(nc, cfg["Layers"], [1, 28, 28]).cuda()
        net.load_state_dict(checkpoint["net"])
        net.eval()

        def predict(images):
            x = images.to(device).unsqueeze(-1).repeat(1, 1, 1, 1, nc["n_steps"])
            return (
                net.forward(x, checkpoint["epoch"] - 1, False)
                .sum(dim=4)
                .squeeze(-1)
                .squeeze(-1)
            )
    else:
        model = {
            "logistic": lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 10)),
            "m5": ModelM5,
            "tssl_cnn": TSSLCNN,
        }[model_name]().to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        def predict(images):
            return model(images.to(device))

    predict.aot_external_data = (
        {"nc": nc, "net": net} if model_name == "tssl" else {"model": model}
    )
    return predict, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("logistic", "m5", "tssl_cnn", "tssl")
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--engine", choices=("eager", "inductor"), default="eager")
    parser.add_argument(
        "--output", type=Path, help="write evaluation to this JSON file"
    )
    args = parser.parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    normalize = args.model in ("tssl", "tssl_cnn")
    transform = transforms.Compose(
        [transforms.ToTensor()]
        + ([transforms.Normalize((0.1307,), (0.3081,))] if normalize else [])
    )
    loader = DataLoader(
        datasets.MNIST(ROOT / "data", train=False, download=False, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
    )
    predict, checkpoint = load_predictor(args.model, args.weights, device)
    predict = inference_engine(predict, args.engine)

    with torch.inference_mode():
        metrics = evaluate_batches(loader, lambda images: predict(images.to(device)))
    result = {
        "model": args.model,
        "engine": args.engine,
        "weights": str(args.weights),
        "best_epoch": checkpoint.get("epoch"),
        "evaluation": "official MNIST test set; selection protocol must be checked in original experiment",
        "test": metrics,
    }
    if args.output:
        atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "accuracy": result["test"]["accuracy"],
                "macro_auc_ovr": result["test"]["macro_auc_ovr"],
                "count": result["test"]["count"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
