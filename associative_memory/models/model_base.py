"""Common model interface, following KAIR naming conventions."""

import torch


class ModelBase:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def init_train(self):
        pass

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
            key: value.detach().clone() for key, value in self.netG.state_dict().items()
        }

    def load_state_dict(self, state):
        return self.netG.load_state_dict(state)
