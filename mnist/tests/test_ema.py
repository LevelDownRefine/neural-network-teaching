import torch
from utils.ema import ParameterEMA


def test_ema_swap_restores_training_parameters():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(1)
    ema = ParameterEMA(model)
    with torch.no_grad(): model.weight.fill_(3)
    ema.update(model)
    with ema.apply(model):
        assert 1 < model.weight.item() < 3
    assert model.weight.item() == 3
