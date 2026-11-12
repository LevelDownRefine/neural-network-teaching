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
        if set(cell.get("metadata", {}).get("tags", [])) & {"lif-imports", "lif-core"}:
            exec("".join(cell["source"]), namespace)
    return namespace["simulate"]


def test_subthreshold_and_critical_current_do_not_spike(simulate):
    for current in (0, 1, 1.5):
        voltage, spikes = simulate(current)
        assert not spikes.any()
        if current == 0:
            assert np.allclose(voltage, -65)


def test_superthreshold_reset_and_refractory(simulate):
    voltage, spikes = simulate(2.0)
    times = np.flatnonzero(spikes)
    assert len(times) > 1
    assert np.all(voltage[times] == -65)
    assert np.all(np.diff(times) > 20)


def test_reaching_threshold_counts_as_spike(simulate):
    voltage, spikes = simulate(1.5, duration_ms=20, dt=20)
    assert spikes.tolist() == [True]
    assert voltage.tolist() == [-65]
