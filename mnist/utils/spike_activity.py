"""Count actual input and first-layer spikes during evaluation."""

import torch


class SpikeActivity:
    def __init__(self, neuron):
        self.neuron = neuron
        self.input_spikes = None
        self.hidden_spikes = None
        self.input_values = 0
        self.hidden_values = 0
        self.images = 0
        self.hook = None

    def __enter__(self):
        self.hook = self.neuron.register_forward_hook(self._count_hidden)
        return self

    def __exit__(self, *args):
        self.hook.remove()

    def observe_input(self, inputs):
        count = inputs.detach().sum(dtype=torch.int64)
        self.input_spikes = (
            count if self.input_spikes is None else self.input_spikes + count
        )
        self.input_values += inputs.numel()
        self.images += inputs.shape[0]

    def _count_hidden(self, module, inputs, output):
        spikes = output[0].detach()
        count = spikes.sum(dtype=torch.int64)
        self.hidden_spikes = (
            count if self.hidden_spikes is None else self.hidden_spikes + count
        )
        self.hidden_values += spikes.numel()

    def result(self):
        def summary(count, values):
            total = int(count.item())
            return {
                "spikes": total,
                "spikes_per_image": total / self.images,
                "spikes_per_neuron_step": total / values,
            }

        if not self.images or not self.hidden_values:
            raise ValueError("no input or hidden spikes were observed")
        return {
            "images": self.images,
            "hidden_layer": "conv_1",
            "input": summary(self.input_spikes, self.input_values),
            "hidden": summary(self.hidden_spikes, self.hidden_values),
        }
