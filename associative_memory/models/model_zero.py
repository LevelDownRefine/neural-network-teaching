from .model_delta import ModelDelta


class ModelZero(ModelDelta):
    """No-write baseline, initialized at zero like the other models."""

    def optimize_parameters(self):
        self._training_pair()
