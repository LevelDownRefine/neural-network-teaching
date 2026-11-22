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


class ZeroTrainer(Trainer):
    """No-write baseline: preserve the supplied model, without an optimizer."""

    def __init__(self, model):
        super().__init__(model, criterion=None, optimizer=None)

    @torch.no_grad()
    def step(self, inputs, target):
        """Validate the pair without updating weights; there is no loss to return."""
        self._validate_loss_inputs(self.model(**inputs), target)


class AttentionTrainer(Trainer):
    """Write explicit K/V with the same result as zero-initialized Hebbian SGD.

    This is a memory writer, not projection training. The supplied attention
    model's eta is the Hebbian learning rate and must stay fixed.
    """

    def __init__(self, model, d=64, m=32, device="cpu"):
        super().__init__(model, criterion=HebbianTrainer.loss, optimizer=None)
        if not isinstance(d, int) or not isinstance(m, int) or min(d, m) <= 0:
            raise ValueError("d and m must be positive integers")
        self.keys = torch.empty(0, d, dtype=torch.float32, device=device)
        self.values = torch.empty(0, m, dtype=torch.float32, device=device)

    @torch.no_grad()
    def step(self, inputs, target):
        """Append one pair and return the same pre-write loss as HebbianTrainer."""
        key = inputs["q"]
        # Validate and compute the loss before changing either stored tensor.
        output = self.model(q=key, k=self.keys, v=self.values)
        loss = self.criterion(output, target)
        if key.ndim != 1:
            raise ValueError("writing requires one key vector")
        if target.device != self.values.device:
            raise ValueError("target and memory must be on the same device")
        keys = torch.cat((self.keys, key.unsqueeze(0)))
        values = torch.cat((self.values, target.unsqueeze(0)))
        self.keys, self.values = keys, values
        return loss

    def memory_weight(self):
        """Materialize the equivalent Hebbian W for comparison or transfer."""
        return self.model.eta * (self.values.T @ self.keys)


def train_epoch(trainer, keys, values, order):
    """Train once per supplied index; repeated indices support replay.

    Inputs are tensors on the model device. The caller owns the random order
    and epoch count; reuse the trainer to preserve optimizer state.
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    for i in order:
        trainer.step({"q": keys[i]}, values[i])
