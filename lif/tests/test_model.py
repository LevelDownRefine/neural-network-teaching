from dataclasses import replace

import numpy as np
import pytest

from lif.model import (
    LIFNeuron,
    SimulationParameters,
    simulate_current,
    simulate_experiment,
)


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


@pytest.mark.parametrize("dt", [0, 0.2, float("nan")])
def test_simulation_rejects_inconsistent_time_step(dt):
    neuron = LIFNeuron(0.1, 20, 10, -65, -50, 2)
    with pytest.raises(ValueError, match="match the neuron"):
        simulate_current(2, 10, dt, neuron)


def test_repeated_experiments_release_temporary_neurons():
    import snntorch as snn

    before = len(snn.SpikingNeuron.instances)
    simulate_experiment(SimulationParameters(duration=1))
    simulate_experiment(SimulationParameters(duration=1))
    with pytest.raises(ValueError):
        simulate_experiment(SimulationParameters(mode="invalid", duration=1))
    assert len(snn.SpikingNeuron.instances) == before


def test_spike_summary_observation_window():
    from lif.model import summarize_spikes

    data = {"time_ms": np.array([1.0, 2.0, 3.0]), "spikes": np.array([0, 1, 1])}
    assert summarize_spikes(data, 4) == {
        "spike_count": 2,
        "average_rate_Hz": 500.0,
        "first_spike_ms": 2.0,
        "mean_ISI_ms": 1.0,
    }
    data["spikes"][:] = 0
    assert summarize_spikes(data, 4)["first_spike_ms"] is None
    assert summarize_spikes(data, 4)["mean_ISI_ms"] is None
