import json
from pathlib import Path

import numpy as np
import pytest
import torch


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
    neuron = simulate.__globals__["LIFNeuron"](20, 20, 10, -65, -65, -50, 0)
    voltage, spike = neuron(1.5)
    assert spike.item() == 1
    assert voltage.item() == -65


def test_equation_parameter_mapping(simulate):
    neuron = simulate.__globals__["LIFNeuron"](0.1, 20, 10, -65, -65, -50, 2)
    assert neuron._relative_voltage(-50) == 15
    assert neuron._decay_factor() == pytest.approx(0.995)
    assert neuron._input_increment(2) == pytest.approx(0.1)


def test_module_accepts_numbers_and_resets(simulate):
    neuron = simulate.__globals__["LIFNeuron"](0.1, 20, 10, -65, -65, -50, 2)
    voltage, spike = neuron(2.0)
    assert spike.item() == 0
    assert voltage.item() == pytest.approx(-64.9)
    neuron.reset_state()
    assert neuron(2.0)[0].item() == pytest.approx(-64.9)


def test_gradient_flows_through_membrane_and_spike(simulate):
    neuron = simulate.__globals__["LIFNeuron"](0.1, 20, 10, -65, -65, -50, 2)
    current = torch.tensor(2.0, requires_grad=True)
    voltage, spike = neuron(current)
    voltage_grad = torch.autograd.grad(voltage, current, retain_graph=True)[0]
    spike_grad = torch.autograd.grad(spike, current)[0]
    assert torch.isfinite(voltage_grad) and voltage_grad > 0
    assert torch.isfinite(spike_grad) and spike_grad > 0


def test_gradient_accumulates_across_time_steps(simulate):
    neuron = simulate.__globals__["LIFNeuron"](0.1, 20, 10, -65, -65, -50, 2)
    current = torch.tensor(2.0, requires_grad=True)
    neuron(current)
    voltage, _ = neuron(current)
    gradient = torch.autograd.grad(voltage, current)[0]
    assert gradient.item() == pytest.approx(0.09975, abs=1e-6)


def test_initial_voltage_is_rest_even_if_reset_differs(simulate):
    neuron = simulate.__globals__["LIFNeuron"](1, 1, 10, -65, -70, -50, 0)
    assert neuron.U.item() == 0
    voltage, spike = neuron(0)
    assert voltage.item() == -65
    assert spike.item() == 0
    voltage, spike = neuron(2)
    assert spike.item() == 1
    assert voltage.item() == -70
    neuron.reset_state()
    assert neuron.U.item() == 0
    assert neuron(0)[0].item() == -65


@pytest.mark.parametrize(
    "dt,tau_m,refractory_ms,V_reset",
    [
        (0, 20, 2, -65),
        (0.1, 0, 2, -65),
        (21, 20, 0, -65),
        (0.1, 20, -1, -65),
        (0.1, 20, 2.05, -65),
        (0.1, 20, 2, -50),
        (float("nan"), 20, 2, -65),
    ],
)
def test_invalid_parameters_are_rejected(simulate, dt, tau_m, refractory_ms, V_reset):
    with pytest.raises(ValueError):
        simulate.__globals__["LIFNeuron"](
            dt, tau_m, 10, -65, V_reset, -50, refractory_ms
        )


def test_rejects_non_scalar_input(simulate):
    neuron = simulate.__globals__["LIFNeuron"](0.1, 20, 10, -65, -65, -50, 2)
    with pytest.raises(ValueError, match="scalar"):
        neuron(torch.tensor([1.0, 2.0]))


def test_reset_state_clears_previous_computation_graph(simulate):
    neuron = simulate.__globals__["LIFNeuron"](0.1, 20, 10, -65, -65, -50, 2)
    neuron(torch.tensor(2.0, requires_grad=True))
    assert neuron.U.grad_fn is not None
    neuron.reset_state()
    assert neuron.U.grad_fn is None
    assert neuron.neuron.mem.grad_fn is None
    assert neuron.remaining.item() == 0


@pytest.mark.parametrize("duration_ms", [0, -1, 0.15, 1e-10, float("nan")])
def test_invalid_duration_is_rejected(simulate, duration_ms):
    with pytest.raises(ValueError):
        simulate(2.0, duration_ms=duration_ms, dt=0.1)
