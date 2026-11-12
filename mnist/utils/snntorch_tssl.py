"""snnTorch equivalent of the TSSL-BP inference forward pass."""

import snntorch as snn
import torch


class TSSLLeaky(snn.Leaky):
    """Immediate hard-reset LIF with explicit state for compilation."""

    def forward(self, input_, membrane):
        membrane = self.beta.clamp(0, 1) * membrane + input_
        spike = self.fire(membrane)
        return spike, membrane * (1 - spike / self.graded_spikes_factor)


def create_neuron(network_config, layer_config):
    return TSSLLeaky(
        beta=1 - 1 / network_config["tau_m"],
        threshold=layer_config["threshold"],
        reset_mechanism="zero",
        reset_delay=False,
    )


def lif_filter(inputs, network_config, layer_config, neuron=None):
    if neuron is None:
        neuron = create_neuron(network_config, layer_config).to(inputs.device)
    membrane = torch.zeros_like(inputs[..., 0])
    synapse = torch.zeros_like(membrane)
    outputs = []
    for t in range(inputs.shape[-1]):
        spike, membrane = neuron(inputs[..., t], membrane)
        synapse = synapse + (spike - synapse) / network_config["tau_s"]
        outputs.append(synapse)
    return torch.stack(outputs, dim=-1)
