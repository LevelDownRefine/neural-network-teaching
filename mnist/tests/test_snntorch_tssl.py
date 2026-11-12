import pytest
import torch

pytest.importorskip("snntorch")

from utils.snntorch_tssl import forward, lif_filter


def test_lif_filter_matches_tsslbp_recurrence():
    inputs = torch.tensor([0.7, 0.7, 0.0, 0.8, 0.6]).reshape(1, 1, 1, 1, 5)
    actual = lif_filter(inputs, {"tau_m": 5, "tau_s": 3}, {"threshold": 1})
    expected = torch.tensor([0, 1 / 3, 2 / 9, 4 / 27, 35 / 81]).reshape_as(actual)
    torch.testing.assert_close(actual, expected)


def test_pooling_does_not_apply_lif():
    class Layer:
        type = "pooling"

        def __call__(self, x):
            return x * 2

    class Network:
        network_config = {"n_steps": 2}
        layers = [Layer()]

    result = forward(Network(), torch.ones(1, 1, 1, 1), lambda x, config: x)
    torch.testing.assert_close(result, torch.tensor([[4.0]]))
