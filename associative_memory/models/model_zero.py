from .model_base import ModelBase


class ModelZero(ModelBase):
    """No-write baseline, initialized at zero like the other models."""

    def optimize_parameters(self):
        self._training_pair()
