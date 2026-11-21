"""Common model interface, following KAIR naming conventions."""

from torch import nn


class ModelBase(nn.Module):
    def forward(self, q):
        """Read from the weight matrix registered by the concrete model."""
        return q @ self.W.T

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
        return super().state_dict()

    def load_state_dict(self, state):
        return super().load_state_dict(state)
