import json
import sys

import evaluate
import pytest
import torch
from torch import nn
from utils import evaluate_checkpoint


def test_logistic_checkpoint_evaluation_uses_common_metrics(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    output = tmp_path / "test.json"
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
        model[1].bias[3] = 5
    torch.save({"model": model.state_dict(), "epoch": 3}, weights)

    class FakeMNIST(torch.utils.data.Dataset):
        def __init__(self, root, train, download, transform):
            assert train is False
            assert download is False
            assert transform is not None

        def __len__(self):
            return 10

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), index

    monkeypatch.setattr(evaluate_checkpoint.datasets, "MNIST", FakeMNIST)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "checkpoint",
            "--model",
            "logistic",
            "--weights",
            str(weights),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    evaluate.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["best_epoch"] == 3
    assert result["test"]["count"] == 10
    assert len(result["test"]["confusion_matrix"]) == 10
    assert result["test"]["confusion_matrix"][3][3] == 1
    assert sum(row[3] for row in result["test"]["confusion_matrix"]) == 10
    assert result["test"]["macro_auc_ovr"] == 0.5


@pytest.mark.parametrize("engine", ["eager", "inductor"])
def test_checkpoint_evaluation_uses_bf16_policy(tmp_path, monkeypatch, capsys, engine):
    weights = tmp_path / "best.pt"
    torch.save(
        {
            "model": nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).state_dict(),
            "epoch": 2,
        },
        weights,
    )

    class FakeMNIST(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), 0

    def fake_evaluate(loader, predict):
        assert torch.is_autocast_enabled("cpu")
        assert predict(torch.zeros(1, 1, 28, 28)).dtype == torch.bfloat16
        return {"accuracy": 1.0, "macro_auc_ovr": 1.0, "count": 1}

    compile_options = []

    def fake_compile(predictor, **kwargs):
        compile_options.append(kwargs)
        return predictor

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(evaluate_checkpoint.datasets, "MNIST", FakeMNIST)
    monkeypatch.setattr(evaluate_checkpoint, "evaluate_batches", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--model",
            "logistic",
            "--weights",
            str(weights),
            "--device",
            "cpu",
            "--precision",
            "bf16",
            "--engine",
            engine,
        ],
    )
    evaluate_checkpoint.main()
    assert json.loads(capsys.readouterr().out)["accuracy"] == 1.0
    assert compile_options == (
        [{"backend": "inductor", "fullgraph": True}] if engine == "inductor" else []
    )
