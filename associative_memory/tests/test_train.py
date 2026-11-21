import pytest
import torch

from associative_memory.models import ModelMemory
from associative_memory.train import DeltaTrainer, HebbianTrainer, Trainer, ZeroTrainer


def test_trainer_keyword_inputs_and_detached_loss():
    class Regression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, **kwargs):
            return self.weight * kwargs["x"] + kwargs["offset"]

    model = Regression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, torch.nn.MSELoss(), optimizer)
    model.eval()
    inputs = {"x": torch.tensor(2.0), "offset": torch.tensor(1.0)}
    loss = trainer.step(inputs, torch.tensor(3.0))
    assert loss.item() == 4.0 and not loss.requires_grad
    assert model.training
    torch.testing.assert_close(model.weight, torch.tensor(0.8))
    trainer.step(inputs, torch.tensor(3.0))
    torch.testing.assert_close(model.weight, torch.tensor(0.96))


@pytest.mark.parametrize("trainer_class", [DeltaTrainer, HebbianTrainer])
@pytest.mark.parametrize("lr", [0.25, 0.5, 1.0])
def test_rule_trainers_match_explicit_updates(trainer_class, lr):
    model = ModelMemory(2, 1)
    model.load_state_dict({"W": torch.tensor([[0.2, -0.3]])})
    expected = model.W.detach().clone()
    trainer = trainer_class(model, lr=lr)
    for k, v in [
        (torch.tensor([0.6, 0.8]), torch.ones(1)),
        (torch.tensor([1.0, 0.0]), -torch.ones(1)),
    ]:
        residual = v - expected @ k if trainer_class is DeltaTrainer else v
        expected += lr * torch.outer(residual, k)
        loss = trainer.step({"q": k}, v)
        assert not loss.requires_grad
        torch.testing.assert_close(model.W, expected)


def test_zero_trainer_never_writes():
    from associative_memory.train import train_epoch

    model = ModelMemory(2, 1)
    trainer = ZeroTrainer(model)
    keys, values = torch.eye(2), torch.ones(2, 1)
    train_epoch(trainer, keys, values, [1, 0, 1])
    torch.testing.assert_close(model.W, torch.zeros(1, 2))
    assert model.W.grad is None and trainer.optimizer is None
    # No-write also means preserving an existing state, not resetting it.
    model.load_state_dict({"W": torch.tensor([[2.0, -3.0]])})
    assert trainer.step({"q": keys[0]}, values[0]) is None
    torch.testing.assert_close(model.W, torch.tensor([[2.0, -3.0]]))
    with pytest.raises(ValueError):
        trainer.step({"q": keys[0]}, torch.ones(2))
