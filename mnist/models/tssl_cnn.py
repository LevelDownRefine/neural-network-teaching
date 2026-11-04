"""Continuous CNN with the same layer sizes and pooling weights as TSSL-BP."""

import torch.nn as nn
import torch.nn.functional as F


class TSSLCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, bias=False)
        self.conv2 = nn.Conv2d(15, 40, 5, bias=False)
        self.fc1 = nn.Linear(640, 300, bias=False)
        self.fc2 = nn.Linear(300, 10, bias=False)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2) * 4.4
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2) * 4.4
        x = F.relu(self.fc1(x.flatten(1)))
        return F.log_softmax(self.fc2(x), dim=1)
