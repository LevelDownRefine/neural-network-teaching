import json

import pytest
import torch
from utils import benchmark_offline


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_offline_benchmark_counts_images_and_handles_no_gpu_power(
    monkeypatch, capsys, precision
):
    class FakeMNIST:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 10

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), index

    clock = iter([10.0, 10.0, 10.2, 10.4, 10.6])
    batches = []
    autocast_states = []

    def predict(batch):
        batches.append(len(batch))
        autocast_states.append(torch.is_autocast_enabled("cpu"))

    monkeypatch.setattr(benchmark_offline.datasets, "MNIST", FakeMNIST)
    monkeypatch.setattr(benchmark_offline.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        benchmark_offline,
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
            "--precision",
            precision,
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["throughput"]["images"] == 8
    assert result["throughput"]["images_per_second"] == pytest.approx(8 / 0.6)
    assert batches == [4, 4, 4]
    assert result["precision"] == precision
    assert autocast_states == [precision == "bf16"] * 3
    assert result["power"]["mean_board_watts"] is None
    assert result["power"]["estimated_board_joules"] is None


def test_bf16_policy_runs_supported_cpu_ops_in_bfloat16():
    from utils.precision import inference_precision

    x = torch.randn(4, 4)
    with inference_precision("bf16", torch.device("cpu")):
        result = x @ x
    assert result.dtype == torch.bfloat16
