import pytest
import torch

pytest.importorskip("snntorch")

from utils.snntorch_tssl import lif_filter


def test_lif_filter_matches_tsslbp_recurrence():
    inputs = torch.tensor([0.7, 0.7, 0.0, 0.8, 0.6]).reshape(1, 1, 1, 1, 5)
    actual = lif_filter(inputs, {"tau_m": 5, "tau_s": 3}, {"threshold": 1})
    expected = torch.tensor([0, 1 / 3, 2 / 9, 4 / 27, 35 / 81]).reshape_as(actual)
    torch.testing.assert_close(actual, expected)


def test_lif_filter_supports_training_gradient():
    inputs = torch.tensor([0.7, 0.7, 0.0, 0.8, 0.6], requires_grad=True).reshape(
        1, 1, 1, 1, 5
    )
    outputs = lif_filter(inputs, {"tau_m": 5, "tau_s": 3}, {"threshold": 1})
    gradient = torch.autograd.grad(outputs.sum(), inputs)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_compilable_neuron_matches_snntorch_forward_and_gradient():
    from snntorch import Leaky
    from utils.snntorch_tssl import create_neuron

    custom = create_neuron({"tau_m": 5}, {"threshold": 1})
    reference = Leaky(beta=0.8, threshold=1, reset_mechanism="zero", reset_delay=False)
    a = torch.tensor([0.7, 0.7, 0.0, 0.8, 0.6], requires_grad=True)
    b = a.detach().clone().requires_grad_()
    mem_a = torch.zeros(())
    mem_b = torch.zeros(())
    spikes_a, spikes_b = [], []
    for index in range(5):
        spk_a, mem_a = custom(a[index], mem_a)
        spk_b, mem_b = reference(b[index], mem_b)
        spikes_a.append(spk_a)
        spikes_b.append(spk_b)
    torch.testing.assert_close(torch.stack(spikes_a), torch.stack(spikes_b))
    grad_a = torch.autograd.grad(torch.stack(spikes_a).sum(), a)[0]
    grad_b = torch.autograd.grad(torch.stack(spikes_b).sum(), b)[0]
    torch.testing.assert_close(grad_a, grad_b)
