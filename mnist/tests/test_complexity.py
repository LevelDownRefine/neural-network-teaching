import json
import sys
from copy import deepcopy

import pytest
import yaml
from models.m5 import ModelM5
from models.tssl_cnn import TSSLCNN
from torch import nn
from utils import complexity
from utils.complexity import ROOT, all_counts, count_dense_layers, count_tssl


def test_counts_for_current_mnist_models():
    counts = all_counts()
    assert (counts["logistic"]["parameters"], counts["logistic"]["macs"]) == (
        7850,
        7840,
    )
    assert (counts["m5"]["parameters"], counts["m5"]["macs"]) == (1128180, 137369600)
    assert (counts["tssl_cnn"]["parameters"], counts["tssl_cnn"]["macs"]) == (
        210375,
        1371000,
    )
    assert (counts["tssl"]["parameters"], counts["tssl"]["macs"]) == (210375, 6855000)
    assert counts["tssl"]["steps"] == 5


def test_tssl_compute_scales_with_time_steps_but_parameters_do_not():
    path = ROOT / "models" / "tssl" / "Networks" / "MNIST_CNN.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    fewer_steps = deepcopy(config)
    fewer_steps["Network"]["n_steps"] = 1
    single = count_tssl(fewer_steps)
    full = count_tssl(config)
    assert full["parameters"] == single["parameters"]
    assert full["macs"] == 5 * single["macs"]
    assert [layer["macs"] for layer in full["layers"]] == [
        1080000,
        4800000,
        960000,
        15000,
    ]


def test_command_reports_floating_point_operations(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["complexity", "tssl"])
    complexity.main()
    result = json.loads(capsys.readouterr().out)["tssl"]
    assert result["flops_approx_per_input"] == 13710000
    assert (
        sum(layer["flops_approx_per_input"] for layer in result["layers"]) == 13710000
    )
    assert "macs" not in result


@pytest.mark.parametrize("size", [24, 28, 32])
def test_spatial_formulas_match_resized_architectures(size):
    a = (size - 4) // 2 - 4
    b = a // 2
    formula_cnn = 750 * (size - 4) ** 2 + 30000 * a**2 + 24000 * b**2 + 6000
    formula_m5 = (
        1600 * (size - 4) ** 2
        + 102400 * (size - 8) ** 2
        + 307200 * (size - 12) ** 2
        + 614400 * (size - 16) ** 2
        + 1027200 * (size - 20) ** 2
    )
    logistic = nn.Sequential(nn.Flatten(), nn.Linear(size * size, 10))
    m5 = ModelM5()
    m5.fc1 = nn.Linear(160 * (size - 20) ** 2, 10, bias=False)
    cnn = TSSLCNN()
    cnn.fc1 = nn.Linear(40 * b * b, 300, bias=False)
    path = ROOT / "models" / "tssl" / "Networks" / "MNIST_CNN.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["Layers"]["FC_1"]["n_inputs"] = 40 * b * b
    assert 2 * count_dense_layers(logistic, size)["macs"] == 20 * size**2
    assert 2 * count_dense_layers(m5, size)["macs"] == formula_m5
    assert 2 * count_dense_layers(cnn, size)["macs"] == formula_cnn
    assert 2 * count_tssl(config, size)["macs"] == 5 * formula_cnn
