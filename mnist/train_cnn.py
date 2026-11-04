"""Train an M5 CNN implementation under our common baseline data split."""

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from data_pipeline.mnist import mnist_loaders
from models.m5 import ModelM5
from models.tssl_cnn import TSSLCNN
from utils.checkpoints import cpu_snapshot, prepare_output, restore_best, save_training
from utils.ema import ParameterEMA
from utils.evaluation import evaluate_batches
from utils.experiment import run_epochs

ROOT = Path(__file__).resolve().parent

CNN_CONFIGS = {
    "m5": {
        "factory": ModelM5,
        "normalization": None,
        "optimizer": torch.optim.Adam,
        "gamma": 0.98,
        "clip": False,
    },
    "tssl_cnn": {
        "factory": TSSLCNN,
        "normalization": ((0.1307,), (0.3081,)),
        "optimizer": torch.optim.AdamW,
        "gamma": 1.0,
        "clip": True,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=tuple(CNN_CONFIGS), default="m5")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="smoke test only")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if args.tag and not all(c.isalnum() or c in "-_" for c in args.tag):
        raise ValueError("invalid tag")
    if (
        args.augment or args.ema or args.max_batches or args.architecture != "m5"
    ) and not args.tag:
        raise ValueError("use --tag to preserve previous results")
    config = CNN_CONFIGS[args.architecture]
    loaders = mnist_loaders(
        args.batch_size,
        args.seed,
        normalization=config["normalization"],
        augmentation=args.augment,
    )
    model = config["factory"]().to(device)
    opt = config["optimizer"](model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config["gamma"])
    ema = ParameterEMA(model) if args.ema else None
    out = (
        ROOT
        / "results"
        / ("cnn_" + args.architecture + "_" + args.tag if args.tag else "cnn_m5_kair")
    )
    prepare_output(out, args.resume)

    def evaluate(loader, collect=False):
        if collect:
            model.eval()
            return evaluate_batches(
                loader, lambda x: model(x.to(device)), args.max_batches
            )
        model.eval()
        n = correct = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                y = y.to(device)
                v = model(x.to(device))
                n += len(y)
                correct += (v.argmax(1) == y).sum().item()
                if args.max_batches and i + 1 >= args.max_batches:
                    break
        return {"accuracy": correct / n, "count": n}

    best = -1.0
    history = []
    start = time.time()
    start_epoch = 0
    best_checkpoint = None
    if args.resume:
        state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if state["config"].get("architecture", "m5") != args.architecture:
            raise ValueError("resume architecture differs")
        for key in ("batch_size", "lr", "seed", "augment", "ema", "max_batches"):
            if state["config"][key] != getattr(args, key):
                raise ValueError("resume configuration differs: " + key)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if ema:
            ema.shadow = {k: v.to(device) for k, v in state["ema_shadow"].items()}
            ema.updates = state["ema_updates"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        best = state["best"]
        history = state["history"]
        start_epoch = state["epoch"]
        best_checkpoint = restore_best(state, out)

    def train_epoch(e):
        model.train()
        n = 0
        total = 0.0
        for i, (x, y) in enumerate(loaders[0]):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = torch.nn.functional.nll_loss(model(x), y)
            loss.backward()
            if config["clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            if config["clip"]:
                with torch.no_grad():
                    for p in model.parameters():
                        p.clamp_(-4, 4)
            if ema:
                ema.update(model)
            n += len(y)
            total += loss.item() * len(y)
            if args.max_batches and i + 1 >= args.max_batches:
                break
        return total / n

    def save_epoch(e, best, history, best_checkpoint):
        (out / "progress.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={e + 1} train_loss={history[-1]['train_loss']:.4f} val_acc={history[-1]['validation_accuracy']:.4f}",
            flush=True,
        )
        scheduler.step()
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": e + 1,
            "best": best,
            "best_checkpoint": best_checkpoint,
            "history": history,
            "config": vars(args),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "ema_shadow": ema.shadow if ema else None,
            "ema_updates": ema.updates if ema else 0,
        }
        save_training(state, out)

    best, history, best_checkpoint = run_epochs(
        start_epoch,
        args.epochs,
        best,
        history,
        best_checkpoint,
        train_epoch=train_epoch,
        validate=lambda e: evaluate(loaders[1]),
        capture=lambda e: {
            "model": cpu_snapshot(model.state_dict()),
            "epoch": e + 1,
            "args": vars(args),
        },
        record=lambda e, training, validation: {
            "epoch": e + 1,
            "train_loss": training,
            "validation_accuracy": validation["accuracy"],
        },
        save_epoch=save_epoch,
        validation_context=(lambda: ema.apply(model)) if ema else nullcontext,
    )
    ck = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    metrics = evaluate(loaders[2], collect=True)
    result = {
        "model": args.architecture,
        "split": {"train": 50000, "validation": 10000, "test": 10000},
        "parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "config": vars(args),
        "data_augmentation": args.augment,
        "ema": args.ema,
        "selection": "validation accuracy",
        "best_epoch": ck["epoch"],
        "validation_best_accuracy": best,
        "test": metrics,
        "history": history,
        "runtime_seconds": time.time() - start,
        "torch": torch.__version__,
        "device": str(device),
        "note": "Independent implementation and common-split adaptation; not an exact paper reproduction.",
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "TEST",
        json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "macro_auc_ovr": metrics["macro_auc_ovr"],
                "count": metrics["count"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
