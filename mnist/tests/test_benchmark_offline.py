import json

import pytest
import torch
from utils import benchmark_common, benchmark_offline


@pytest.mark.parametrize("engine", ["eager", "inductor"])
def test_offline_benchmark_counts_images_and_handles_no_gpu_power(
    monkeypatch, capsys, engine
):
    class FakeMNIST:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 10

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), index

    clock = iter([9.0, 9.5, 10.0, 10.0, 10.2, 10.4, 10.6])
    batches = []
    compile_options = []

    def fake_compile(predictor, **kwargs):
        compile_options.append(kwargs)
        return predictor

    monkeypatch.setattr(torch, "compile", fake_compile)

    def predict(batch):
        batches.append(len(batch))

    monkeypatch.setattr(benchmark_common.datasets, "MNIST", FakeMNIST)
    monkeypatch.setattr(benchmark_common.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        benchmark_common,
        "load_predictor",
        lambda model, weights, device: (
            predict,
            {"epoch": 3},
        ),
    )
    benchmark_offline.main(
        [
            "--model",
            "logistic",
            "--weights",
            "unused.pt",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--warmup",
            "1",
            "--seconds",
            "0.4",
            "--engine",
            engine,
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["throughput"]["images"] == 8
    assert result["throughput"]["images_per_second"] == pytest.approx(8 / 0.6)
    assert batches == [4, 4, 4]
    assert result["engine"] == engine
    assert compile_options == (
        [{"backend": "inductor", "fullgraph": True}] if engine == "inductor" else []
    )
    assert result["power"]["mean_board_watts"] is None
    assert result["power"]["estimated_board_joules"] is None
