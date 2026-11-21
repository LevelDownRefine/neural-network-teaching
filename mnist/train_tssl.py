"""Train the TSSL-architecture SNN using snnTorch surrogate gradients."""

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "models" / "tssl"
sys.path.insert(0, str(SRC))
import numpy as np
import torch
import yaml
from data_pipeline.mnist import mnist_loaders
from encoding import encode_input
from utils.checkpoints import prepare_output, save_training, restore_best, cpu_snapshot
from utils.cli import positive_int
from utils.ema import ParameterEMA
from utils.experiment import run_epochs
from utils.evaluation import evaluate_batches
from utils.io import atomic_write_json
from utils.inference_timing import mean_batch_latency
from utils.spike_activity import SpikeActivity
import cnns
import functions.loss_f as loss_f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=positive_int, default=100)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--hidden-size", type=int, default=300)
    p.add_argument(
        "--encoding", choices=("repeat", "rate", "latency", "phase"), default="repeat"
    )
    p.add_argument("--steps", type=positive_int)
    p.add_argument("--phase-period", type=positive_int, default=20)
    p.add_argument("--latency-warmup", type=positive_int, default=3)
    p.add_argument("--latency-repeats", type=positive_int, default=10)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--max-batches", type=int, default=0, help="smoke test only")
    p.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="run a small CPU smoke test",
    )
    p.add_argument("--tag", default="")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.steps is not None and a.steps < 2:
        p.error("--steps must be at least 2")
    if (a.encoding != "repeat" or a.steps is not None) and not a.tag:
        p.error("use --tag for encoding experiments")
    if (a.augment or a.ema) and not a.tag:
        p.error("use --tag for modified experiments")
    if a.cpu_smoke:
        if not a.max_batches:
            raise ValueError("--cpu-smoke requires --max-batches")
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
        torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    elif not torch.cuda.is_available():
        raise RuntimeError("TSSL source requires CUDA")
    device = "cpu" if a.cpu_smoke else "cuda"
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    cfg = yaml.safe_load((SRC / "Networks" / "MNIST_CNN.yaml").read_text())
    nc = cfg["Network"]
    nc["batch_size"] = a.batch_size
    if a.steps is not None:
        nc["n_steps"] = a.steps
    cfg["Layers"]["FC_1"]["n_outputs"] = a.hidden_size
    cfg["Layers"]["output"]["n_inputs"] = a.hidden_size
    loaders = mnist_loaders(
        a.batch_size,
        a.seed,
        normalization=((0.1307,), (0.3081,)) if a.encoding == "repeat" else None,
        augmentation=a.augment,
    )
    net = cnns.Network(nc, cfg["Layers"], [1, 28, 28]).cuda()
    err = loss_f.SpikeLoss(nc).cuda()
    opt = torch.optim.AdamW(net.get_parameters(), lr=nc["lr"], betas=(0.9, 0.999))
    ema = ParameterEMA(net) if a.ema else None
    steps = nc["n_steps"]
    desired = torch.ones(steps, device=device)
    desired[0] = 0
    desired = desired.view(1, 1, 1, 1, steps)
    desired = loss_f.psp(desired, nc).view(1, 1, 1, steps)
    train_generator = torch.Generator(device=device).manual_seed(a.seed + 1)

    def encode(images, generator):
        kwargs = {"generator": generator}
        if a.encoding == "phase":
            kwargs["period"] = a.phase_period
        return encode_input(images, steps, a.encoding, **kwargs)

    if a.tag and not all(c.isalnum() or c in "-_" for c in a.tag):
        raise ValueError("invalid tag")
    if a.hidden_size != 300 and not a.tag:
        raise ValueError("use --tag to preserve previous results")
    out = (
        ROOT
        / "results"
        / (
            ("snntorch_smoke" if a.max_batches else "snntorch")
            + ("_" + a.tag if a.tag else "")
        )
    )
    prepare_output(out, a.resume)

    def run(loader, epoch, training, collect=False, activity=None):
        generator = (
            train_generator
            if training
            else torch.Generator(device=device).manual_seed(a.seed + 2)
        )
        if collect:
            net.eval()

            def predict(x):
                x = encode(x.cuda(), generator)
                if activity is not None:
                    activity.observe_input(x)
                return net.forward(x, epoch, False).sum(dim=4).squeeze(-1).squeeze(-1)

            return evaluate_batches(loader, predict, a.max_batches)
        net.train() if training else net.eval()
        n = correct = 0
        losses = 0.0
        for idx, (x, y) in enumerate(loader):
            x = encode(x.cuda(), generator)
            y = y.cuda()
            with torch.set_grad_enabled(training):
                pred = net.forward(x, epoch, training)
                if training:
                    target = torch.zeros((len(y), 10, 1, 1, steps), device=device)
                    target[torch.arange(len(y), device=device), y] = desired
                    loss = err.spike_kernel(pred, target, nc)
                    if a.encoding != "repeat":
                        loss = loss / steps
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.get_parameters(), 1)
                    opt.step()
                    net.weight_clipper()
                    if ema:
                        ema.update(net)
                    losses += loss.item()
            scores = pred.sum(dim=4).squeeze(-1).squeeze(-1)
            correct += (scores.argmax(1) == y).sum().item()
            n += len(y)
            if a.max_batches and idx + 1 >= a.max_batches:
                break
        return {
            "accuracy": correct / n,
            "count": n,
            "loss": losses / n if training else None,
        }

    best = -1
    history = []
    start_epoch = 0
    best_checkpoint = None
    if a.resume:
        state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if state["config"].get("hidden_size", 300) != a.hidden_size:
            raise ValueError("resume configuration differs: hidden_size")
        for key in ("augment", "ema"):
            if state["config"].get(key, False) != getattr(a, key):
                raise ValueError("resume configuration differs: " + key)
        for key in ("batch_size", "seed", "max_batches", "cpu_smoke"):
            if state["config"][key] != getattr(a, key):
                raise ValueError("resume configuration differs: " + key)
        for key, default in (
            ("encoding", "repeat"),
            ("steps", None),
            ("phase_period", 20),
        ):
            if state["config"].get(key, default) != getattr(a, key):
                raise ValueError("resume configuration differs: " + key)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["optimizer"])
        if ema:
            ema.shadow = {k: v.to(device) for k, v in state["ema_shadow"].items()}
            ema.updates = state["ema_updates"]
        best = state["best"]
        history = state["history"]
        start_epoch = state["epoch"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if not a.cpu_smoke:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "encoding_rng" in state:
            train_generator.set_state(state["encoding_rng"])
        best_checkpoint = restore_best(state, out)

    def save_epoch(e, best, history, best_checkpoint):
        state = {
            "net": net.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": e + 1,
            "best": best,
            "best_checkpoint": best_checkpoint,
            "history": history,
            "config": vars(a),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if not a.cpu_smoke else [],
            "encoding_rng": train_generator.get_state(),
            "ema_shadow": ema.shadow if ema else None,
            "ema_updates": ema.updates if ema else 0,
        }
        save_training(state, out)
        (out / "progress.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={e + 1} train={history[-1]['train']['accuracy']:.4f} val={history[-1]['validation']['accuracy']:.4f}",
            flush=True,
        )

    def timed_run(loader, epoch, training):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = run(loader, epoch, training)
        if device == "cuda":
            torch.cuda.synchronize()
        result["seconds"] = time.perf_counter() - start
        return result

    best, history, best_checkpoint = run_epochs(
        start_epoch,
        a.epochs,
        best,
        history,
        best_checkpoint,
        train_epoch=lambda e: timed_run(loaders[0], e, True),
        validate=lambda e: timed_run(loaders[1], e, False),
        capture=lambda e: {"net": cpu_snapshot(net.state_dict()), "epoch": e + 1},
        record=lambda e, tr, va: {
            "epoch": e + 1,
            "train": tr,
            "validation": va,
        },
        save_epoch=save_epoch,
        validation_context=(lambda: ema.apply(net)) if ema else nullcontext,
    )
    ck = torch.load(out / "best.pt", weights_only=True)
    net.load_state_dict(ck["net"])
    activity_result = None
    latency_result = None
    if a.encoding == "repeat":
        te = run(loaders[2], ck["epoch"], False, collect=True)
    else:
        with SpikeActivity(net.layers[0]._snn_neuron) as activity:
            te = run(loaders[2], ck["epoch"], False, collect=True, activity=activity)
        activity_result = activity.result()
        batch = next(iter(loaders[2]))[0].to(device)
        generator = torch.Generator(device=device).manual_seed(a.seed + 3)
        fixed_input = encode(batch, generator)

        def forward_encoded(x):
            return net.forward(x, ck["epoch"], False)

        def forward_raw(x):
            return forward_encoded(encode(x, generator))

        latency_result = {
            "model_only": mean_batch_latency(
                forward_encoded, fixed_input, a.latency_warmup, a.latency_repeats
            ),
            "encoding_and_model": mean_batch_latency(
                forward_raw, batch, a.latency_warmup, a.latency_repeats
            ),
            "warmup": a.latency_warmup,
            "device_transfer_included": False,
        }
    atomic_write_json(
        out / "metrics.json",
        {
            "model": "tssl",
            "split": {
                "train": 50000,
                "validation": 10000,
                "test": 10000,
            },
            "config": vars(a),
            "parameter_count": sum(
                p.numel() for p in net.get_parameters() if p.requires_grad
            ),
            "selection": "validation accuracy",
            "test": te,
            "spike_activity": activity_result,
            "latency": latency_result,
            "measurement_note": "Latency uses a preloaded test batch, synchronized per iteration; model_only excludes encoding, encoding_and_model includes encoding; data loading and transfer excluded.",
            "best_epoch": ck["epoch"],
            "history": history,
            "source": "TSSL architecture and loss with snnTorch surrogate gradients",
            "device": "CPU smoke only"
            if a.cpu_smoke
            else torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    print("TEST", te, flush=True)


if __name__ == "__main__":
    main()
