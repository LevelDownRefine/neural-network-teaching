"""Measure synchronized mean inference latency for a preloaded batch."""

import statistics
import time

import torch


def mean_batch_latency(predict, batch, warmup, repeats):
    def sync():
        if batch.device.type == "cuda":
            torch.cuda.synchronize(batch.device)

    with torch.inference_mode():
        for _ in range(warmup):
            predict(batch)
        sync()
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            predict(batch)
            sync()
            durations.append((time.perf_counter() - start) * 1000)
    return {
        "mean_batch_ms": statistics.mean(durations),
        "batch_size": len(batch),
        "repeats": repeats,
    }
