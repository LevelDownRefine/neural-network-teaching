import torch
from utils.inference_timing import mean_batch_latency
from utils.spike_activity import SpikeActivity


class Neuron(torch.nn.Module):
    def forward(self, spikes):
        return spikes, torch.zeros_like(spikes)


def test_spike_activity_counts_inputs_and_raw_hidden_outputs():
    neuron = Neuron()
    inputs = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    with SpikeActivity(neuron) as activity:
        activity.observe_input(inputs)
        neuron(torch.tensor([[1.0, 0.0, 1.0]]))
        neuron(torch.tensor([[0.0, 0.0, 1.0]]))
    result = activity.result()
    assert result["images"] == 1
    assert result["input"] == {
        "spikes": 2,
        "spikes_per_image": 2,
        "spikes_per_neuron_step": 0.5,
    }
    assert result["hidden"] == {
        "spikes": 3,
        "spikes_per_image": 3,
        "spikes_per_neuron_step": 0.5,
    }
    assert not neuron._forward_hooks


def test_timing_runs_warmup_and_repeated_measurements():
    calls = []
    batch = torch.zeros(4, 1)
    result = mean_batch_latency(lambda x: calls.append(x), batch, warmup=2, repeats=3)
    assert len(calls) == 5
    assert result["batch_size"] == 4
    assert result["repeats"] == 3
    assert result["mean_batch_ms"] >= 0
