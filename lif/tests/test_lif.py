import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def simulate():
    notebook = json.loads(
        (Path(__file__).parents[1] / "task1_lif.ipynb").read_text(encoding="utf-8")
    )
    namespace = {}
    for cell in notebook["cells"]:
        if set(cell.get("metadata", {}).get("tags", [])) & {
            "lif-imports",
            "lif-defaults",
            "lif-core",
        }:
            exec("".join(cell["source"]), namespace)
    return namespace["simulate"]


def test_subthreshold_and_critical_current_do_not_spike(simulate):
    for current in (0, 1, 1.5):
        voltage, spikes = simulate(current, duration_ms=2000, dt=0.1)
        assert not spikes.any()
        if current == 0:
            assert np.allclose(voltage, -65)


def test_superthreshold_reset_and_refractory(simulate):
    voltage, spikes = simulate(2.0, duration_ms=2000, dt=0.1)
    times = np.flatnonzero(spikes)
    assert len(times) > 1
    assert np.all(voltage[times] == -65)
    assert np.all(np.diff(times) > 20)


def test_reaching_threshold_counts_as_spike(simulate):
    voltage, spikes = simulate(1.5, duration_ms=20, dt=20)
    assert spikes.tolist() == [True]
    assert voltage.tolist() == [-65]


def test_equation_parameter_mapping(simulate):
    functions = simulate.__globals__
    assert functions["relative_voltage"](-50, -65) == 15
    assert functions["decay_factor"](0.1, 20) == pytest.approx(0.995)
    assert functions["input_increment"](0.1, 20, 10, 2) == pytest.approx(0.1)


def test_wrapper_accepts_and_returns_numbers(simulate):
    functions = simulate.__globals__
    neuron = functions["create_lif"](0.1, 20, 15)
    spike, voltage = functions["lif_step"](neuron, 0.0, 0.1)
    assert spike is False
    assert voltage == pytest.approx(0.1)
