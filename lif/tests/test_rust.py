from dataclasses import replace

import numpy as np
import pytest

native = pytest.importorskip("lif_native")

from lif.model import (  # noqa: E402
    LIFNeuron,
    SimulationParameters,
    simulate_current,
    simulate_experiment,
)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"current": 0},
        {"current": 1.5},
        {"current": 4},
        {"tau": 5},
        {"tau": 50},
        {"threshold": -60},
        {"threshold": -45},
        {"refractory": 0},
        {"refractory": 10},
        {"mode": "sinusoidal"},
        {"mode": "sinusoidal", "current": 0, "amplitude": 3},
        {"mode": "sinusoidal", "frequency": 50, "tau": 7},
    ],
)
def test_rust_matches_snntorch(change):
    parameters = replace(SimulationParameters(duration=200), **change)
    reference = simulate_experiment(parameters)
    actual = simulate_experiment(parameters, backend="rust")
    for key in reference:
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
    assert actual["voltage"].dtype == np.float32


@pytest.mark.parametrize("mode,count", [("constant", 67), ("sinusoidal", 60)])
def test_full_length_reference(mode, count):
    p = SimulationParameters(mode=mode)
    reference = simulate_experiment(p)
    actual = simulate_experiment(p, backend="rust")
    np.testing.assert_array_equal(actual["spikes"], reference["spikes"])
    np.testing.assert_array_equal(actual["voltage"], reference["voltage"])
    assert int(actual["spikes"].sum()) == count


@pytest.mark.parametrize("seed", range(10))
def test_random_current_matches_reference(seed):
    rng = np.random.default_rng(seed)
    current = rng.uniform(-2, 5, 1000).astype(np.float32)
    neuron = LIFNeuron(0.1, 20, 10, -65, -50, 2)
    voltage, spikes = simulate_current(current, 100, 0.1, neuron)
    actual_voltage, actual_spikes = native.simulate(
        current.tolist(), 0.995, 0.05, -65, 15, 20
    )
    np.testing.assert_array_equal(np.asarray(actual_voltage, dtype=np.float32), voltage)
    np.testing.assert_array_equal(np.frombuffer(actual_spikes, dtype=np.uint8), spikes)


@pytest.mark.parametrize("current", [[], [float("nan")], [float("inf")]])
def test_native_rejects_invalid_current(current):
    with pytest.raises(ValueError):
        native.simulate(current, 0.995, 0.05, -65, 15, 20)


def test_exact_threshold_and_refractory():
    voltage, spikes = native.simulate([1.5] * 4, 0, 10, -65, 15, 2)
    assert voltage == [-65] * 4
    assert spikes == bytes([1, 0, 0, 1])


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"mode": "sinusoidal"},
        {"mode": "sinusoidal", "frequency": 50, "tau": 7},
        {"current": 0},
        {"threshold": -55},
        {"refractory": 0},
    ],
)
def test_native_frontend_pipeline_matches_reference(change):
    from dataclasses import asdict

    p = replace(SimulationParameters(duration=200), **change)
    fields = asdict(p)
    fields["sinusoidal"] = fields.pop("mode") == "sinusoidal"
    fields.pop("rest")
    fields.pop("dt")
    current, voltage, spikes, rate = native.run_experiment(fields)
    reference = simulate_experiment(p)
    np.testing.assert_array_equal(
        np.asarray(current, dtype=np.float32), reference["current"].astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(voltage, dtype=np.float32), reference["voltage"]
    )
    np.testing.assert_array_equal(
        np.frombuffer(spikes, dtype=np.uint8), reference["spikes"]
    )
    np.testing.assert_array_equal(rate, reference["rate"])
