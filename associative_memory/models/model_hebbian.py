import numpy as np

from .model_delta import ModelDelta


class ModelHebbian(ModelDelta):
    def optimize_parameters(self):
        k, v = self._training_pair()
        self.W += self.eta * np.outer(v, k)
