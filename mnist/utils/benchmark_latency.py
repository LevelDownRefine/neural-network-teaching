"""Measure steady-state model batch latency on CUDA."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from utils.cli import positive_int
from utils.evaluate_checkpoint import load_predictor
from utils.inference import inference_engine
from utils.io import atomic_write_json


def percentile(samples, fraction):
    return sorted(samples)[math.ceil(len(samples) * fraction) - 1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("logistic", "m5", "tssl_cnn", "tssl")
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--engine", choices=("eager", "inductor"), default="inductor")
    parser.add_argument("--warmup", type=positive_int, default=50)
    parser.add_argument("--samples", type=positive_int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.manual_seed(42)
    device = torch.device("cuda")
    predict, checkpoint = load_predictor(args.model, args.weights, device)
    x = torch.randn(args.batch_size, 1, 28, 28, device=device)
    samples = []
    with torch.inference_mode():
        predict = inference_engine(predict, args.engine)
        for _ in range(args.warmup):
            predict(x)
        torch.cuda.synchronize()
        for _ in range(args.samples):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            predict(x)
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    result = {
        "model": args.model,
        "weights": str(args.weights),
        "best_epoch": checkpoint.get("epoch"),
        "batch_size": args.batch_size,
        "engine": args.engine,
        "warmup": args.warmup,
        "samples": len(samples),
        "p90_ms": percentile(samples, 0.90),
        "p99_ms": percentile(samples, 0.99),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "timing": "host wall time through CUDA synchronization; random inputs preloaded on GPU",
    }
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
