"""Training helpers shared by associative-memory experiments."""

import torch

from .losses import delta_loss, hebbian_loss


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


class DeltaTrainer(Trainer):
    """Delta rule through half squared error and plain SGD."""

    def __init__(self, model, lr=0.5):
        super().__init__(model, delta_loss, torch.optim.SGD(model.parameters(), lr=lr))


class HebbianTrainer(Trainer):
    """Hebbian rule through negative correlation and plain SGD."""

    def __init__(self, model, lr=0.5):
        super().__init__(
            model, hebbian_loss, torch.optim.SGD(model.parameters(), lr=lr)
        )


def train_epoch(trainer, keys, values, order):
    """Train once per supplied index; repeated indices support replay.

    Inputs are tensors on the model device. The caller owns the random order
    and epoch count; reuse the trainer to preserve optimizer state.
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    for i in order:
        trainer.step({"q": keys[i]}, values[i])
