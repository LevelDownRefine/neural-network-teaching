"""Exponential average of model parameters with temporary evaluation swap."""
from contextlib import contextmanager
import torch


class ParameterEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.detach(), alpha=1-decay)

    @contextmanager
    def apply(self, model):
        parameters = dict(model.named_parameters())
        original = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items(): parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in original.items(): parameters[name].copy_(value)
