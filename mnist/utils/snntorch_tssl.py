"""snnTorch equivalent of the TSSL-BP inference forward pass."""

import torch


def lif_filter(inputs, network_config, layer_config):
    import snntorch as snn

    neuron = snn.Leaky(
        beta=1 - 1 / network_config["tau_m"],
        threshold=layer_config["threshold"],
        reset_mechanism="zero",
        reset_delay=False,
    ).to(inputs.device)
    membrane = torch.zeros_like(inputs[..., 0])
    synapse = torch.zeros_like(membrane)
    outputs = []
    for t in range(inputs.shape[-1]):
        spike, membrane = neuron(inputs[..., t], membrane)
        synapse = synapse + (spike - synapse) / network_config["tau_s"]
        outputs.append(synapse)
    return torch.stack(outputs, dim=-1)


def forward(network, images, psp):
    steps = network.network_config["n_steps"]
    spikes = psp(images.unsqueeze(-1).repeat(1, 1, 1, 1, steps), network.network_config)
    for layer in network.layers:
        if layer.type == "dropout":
            continue
        spikes = layer(spikes)
        if layer.type != "pooling":
            spikes = lif_filter(spikes, network.network_config, layer.layer_config)
    return spikes.sum(dim=4).squeeze(-1).squeeze(-1)
