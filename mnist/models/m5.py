"""M5-style single CNN, compatible with the saved MNIST checkpoint.

Architecture: An et al., https://arxiv.org/abs/2008.10400.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ModelM5(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (1, 32, 64, 96, 128, 160)
        for index, (input_channels, output_channels) in enumerate(zip(channels, channels[1:]), start=1):
            setattr(self, f'conv{index}', nn.Conv2d(input_channels, output_channels, 5, bias=False))
            setattr(self, f'conv{index}_bn', nn.BatchNorm2d(output_channels))
        self.fc1 = nn.Linear(8 * 8 * channels[-1], 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, image):
        features = image.mul(2).sub(1)
        for index in range(1, 6):
            features = F.relu(getattr(self, f'conv{index}_bn')(getattr(self, f'conv{index}')(features)))
        features = features.permute(0, 2, 3, 1).flatten(1)
        return self.fc1_bn(self.fc1(features))

    def forward(self, image):
        return F.log_softmax(self.get_logits(image), dim=1)
