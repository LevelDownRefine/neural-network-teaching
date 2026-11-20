import numpy as np

from .model_base import ModelBase


class ModelDelta(ModelBase):
    def optimize_parameters(self):
        k, v = self._training_pair()
        error = v - self.W @ k
        self.W += self.eta * np.outer(error, k)
