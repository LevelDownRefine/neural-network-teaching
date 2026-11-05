"""Benchmark offline MNIST throughput and GPU board power."""

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
from torchvision import datasets, transforms
from utils.cli import positive_int
from utils.evaluate_checkpoint import load_predictor
from utils.inference import inference_engine
from utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parent.parent


def sample_power(stop, samples, interval, gpu_id):
    """Collect device-wide board power, without blocking the inference thread."""
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_id}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            samples.append((time.perf_counter(), float(result.stdout.strip())))
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        stop.wait(interval)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, choices=("logistic", "m5", "tssl_cnn", "tssl")
    )
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=positive_int, default=512)
    p.add_argument("--engine", choices=("eager", "inductor"), default="eager")
    p.add_argument("--warmup", type=positive_int, default=20)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--power-interval", type=float, default=1.0)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if not all(math.isfinite(x) and x > 0 for x in (args.seconds, args.power_interval)):
        p.error("--seconds and --power-interval must be positive")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    transform = [transforms.ToTensor()]
    if args.model in ("tssl", "tssl_cnn"):
        transform.append(transforms.Normalize((0.1307,), (0.3081,)))
    data = datasets.MNIST(
        ROOT / "data",
        train=False,
        download=False,
        transform=transforms.Compose(transform),
    )
    if args.batch_size > len(data):
        p.error("batch size exceeds test set size")
    batch = torch.stack([data[i][0] for i in range(args.batch_size)]).to(device)
    predict, checkpoint = load_predictor(args.model, args.weights, device)
    gpu_id = (
        "GPU-" + str(torch.cuda.get_device_properties(device).uuid)
        if device.type == "cuda"
        else None
    )
    samples = []
    stop = threading.Event()
    with torch.inference_mode():
        setup_start = time.perf_counter()
        predict = inference_engine(predict, args.engine)
        for _ in range(args.warmup):
            predict(batch)
        synchronize(device)
        setup_seconds = time.perf_counter() - setup_start
        sampler = None
        if device.type == "cuda":
            sampler = threading.Thread(
                target=sample_power,
                args=(stop, samples, args.power_interval, gpu_id),
                daemon=True,
            )
            sampler.start()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < args.seconds:
            predict(batch)
            synchronize(device)
            count += args.batch_size
        end = time.perf_counter()
        stop.set()
        if sampler:
            sampler.join(timeout=6)
    elapsed = end - start
    watts = [power for timestamp, power in samples if start <= timestamp <= end]
    mean_power = statistics.mean(watts) if len(watts) >= 2 else None
    result = {
        "mode": "offline",
        "engine": args.engine,
        "fullgraph": args.engine == "inductor",
        "warmup_and_compile_seconds": setup_seconds,
        "model": args.model,
        "weights": str(args.weights),
        "best_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_uuid": gpu_id,
        "torch": torch.__version__,
        "input": "MNIST test images, preloaded on device; preprocessing and transfer excluded",
        "warmup_batches_per_shape": args.warmup,
        "throughput": {
            "batch_size": args.batch_size,
            "images": count,
            "seconds": elapsed,
            "images_per_second": count / elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        },
        "power": {
            "samples": len(watts),
            "readings": [
                {"seconds": timestamp - start, "watts": power}
                for timestamp, power in samples
                if start <= timestamp <= end
            ],
            "interval_seconds": args.power_interval,
            "mean_board_watts": mean_power,
            "estimated_board_joules": mean_power * elapsed
            if mean_power is not None
            else None,
            "estimated_board_joules_per_image": mean_power * elapsed / count
            if mean_power is not None
            else None,
        },
        "power_note": "Approximate total GPU board power including idle/background load; sampled by nvidia-smi during throughput, not whole-system or model-only energy.",
        "timing_note": "Host wall time with CUDA synchronization; includes prediction and time-step expansion, excludes data loading and transfers.",
    }
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
