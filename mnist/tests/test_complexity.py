import json
import sys
from copy import deepcopy

import yaml
from utils import complexity
from utils.complexity import ROOT, all_counts, count_tssl


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
