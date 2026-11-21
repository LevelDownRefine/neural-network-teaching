"""Training helpers shared by associative-memory experiments."""

import torch


class Trainer:
    """Compose a model, objective and optimizer without changing their APIs."""

    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def step(self, inputs, target):
        """Train on keyword inputs and return the detached pre-update loss."""
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(**inputs), target)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    @staticmethod
    def _validate_loss_inputs(pred, target):
        for tensor in (pred, target):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("loss inputs must be torch tensors")
            assert tensor.dtype == torch.float32, "loss inputs must be float32"
            if tensor.ndim != 1 or not tensor.isfinite().all():
                raise ValueError("loss requires finite vectors for one training pair")
        if pred.shape != target.shape:
            raise ValueError("prediction and target shapes must match")


class DeltaTrainer(Trainer):
    """Delta rule through half squared error and plain SGD."""

    def __init__(self, model, lr=0.5):
        super().__init__(model, self.loss, torch.optim.SGD(model.parameters(), lr=lr))

    @staticmethod
    def loss(pred, target):
        """Half squared error, summed over output coordinates."""
        Trainer._validate_loss_inputs(pred, target)
        return 0.5 * (pred - target).square().sum()


class HebbianTrainer(Trainer):
    """Hebbian rule through negative correlation and plain SGD."""

    def __init__(self, model, lr=0.5):
        super().__init__(model, self.loss, torch.optim.SGD(model.parameters(), lr=lr))

    @staticmethod
    def loss(pred, target):
        """Negative correlation objective; unbounded, not an error metric."""
        Trainer._validate_loss_inputs(pred, target)
        return -(pred * target).sum()


def train_epoch(trainer, keys, values, order):
    """Train once per supplied index; repeated indices support replay.

    Inputs are tensors on the model device. The caller owns the random order
    and epoch count; reuse the trainer to preserve optimizer state.
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    for i in order:
        trainer.step({"q": keys[i]}, values[i])
