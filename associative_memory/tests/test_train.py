import torch

from associative_memory.train import Trainer


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
