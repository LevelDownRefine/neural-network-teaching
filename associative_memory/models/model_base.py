"""Common model interface, following KAIR naming conventions."""

from torch import nn


class ModelBase(nn.Module):
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
