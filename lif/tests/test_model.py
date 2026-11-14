from dataclasses import replace

import numpy as np
import pytest

from lif.model import SimulationParameters, simulate_experiment


def test_input_waveform_and_time_units():
    parameters = SimulationParameters(mode="sinusoidal", duration=200)
    result = simulate_experiment(parameters)
    expected = 2 + 0.8 * np.sin(2 * np.pi * 5 * result["input_time_ms"] / 1000)
    np.testing.assert_allclose(result["current"], expected)
    assert result["time_ms"][-1] == pytest.approx(200)
    assert np.isfinite(result["rate"]).all()


@pytest.mark.parametrize(
    "change",
    [
        {"current": float("nan")},
        {"mode": "unknown"},
        {"amplitude": -1},
        {"frequency": -1},
        {"duration": 0},
    ],
)
def test_invalid_experiment_parameters(change):
    with pytest.raises(ValueError):
        simulate_experiment(replace(SimulationParameters(), **change))
