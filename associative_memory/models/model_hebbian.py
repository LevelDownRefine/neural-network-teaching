import numpy as np

from .model_base import ModelBase


class ModelHebbian(ModelBase):
    def optimize_parameters(self):
        k, v = self._training_pair()
        self.W += self.eta * np.outer(v, k)
