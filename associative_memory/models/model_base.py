"""Common model interface, following KAIR naming conventions."""

import torch


class ModelBase(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def test(self):
        pass

    def current_results(self):
        pass

    def reset(self):
        pass

    def state_dict(self):
        return {
            key: value.detach().clone() for key, value in super().state_dict().items()
        }

    def load_state_dict(self, state):
        return super().load_state_dict(state)
