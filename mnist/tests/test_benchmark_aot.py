import builtins
import json

import pytest
import torch
from utils import benchmark_common, benchmark_power


class TinyDataset:
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), 0


def test_compiled_artifact_and_manual_start(tmp_path, monkeypatch):
    events = []
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    artifact = tmp_path / "artifact"

    def predict(batch):
        events.append("predict")
        return batch

    predict.aot_external_data = {"model": object()}
    monkeypatch.setattr(benchmark_common.datasets, "MNIST", TinyDataset)
    monkeypatch.setattr(benchmark_common, "load_predictor", lambda *args: (predict, {}))

    class Compiled:
        def aot_compile(self, inputs):
            assert inputs[0][0].shape == (1, 1, 28, 28)
            return self

        def save_compiled_function(self, path, external_data):
            assert external_data == predict.aot_external_data
            from pathlib import Path

            Path(path).write_bytes(b"compiled")

    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: Compiled())
    base = [
        "--model",
        "logistic",
        "--weights",
        str(weights),
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--engine",
        "inductor",
    ]
    benchmark_power.main(base + ["--save-compiled", str(artifact)])
    assert json.loads((artifact / "metadata.json").read_text())["batch_size"] == 1

    def load(file, external_data):
        assert file.read() == b"compiled"
        assert external_data == predict.aot_external_data
        return predict

    monkeypatch.setattr(torch.compiler, "load_compiled_function", load)
    monkeypatch.setattr(builtins, "input", lambda prompt: events.append("input"))
    output = tmp_path / "result.json"
    benchmark_power.main(
        base
        + [
            "--load-compiled",
            str(artifact),
            "--warmup",
            "1",
            "--seconds",
            "0.001",
            "--output",
            str(output),
        ]
    )
    assert events[:2] == ["predict", "input"]
    assert len(events) > 2
    result = json.loads(output.read_text())
    assert result["compiled_artifact"] == str(artifact)
    assert result["throughput"]["seconds"] >= 0.001
    assert result["throughput"]["images"] > 0
    weights.write_bytes(b"different")
    with pytest.raises(ValueError, match="metadata"):
        benchmark_power.main(base + ["--load-compiled", str(artifact)])
