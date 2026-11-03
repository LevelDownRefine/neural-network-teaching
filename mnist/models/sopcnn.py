"""SOPCNN-style architecture based on junaidaliop/MNIST-SOPCNN (MIT)."""
import torch.nn as nn
import torch.nn.functional as F


class ModelSOPCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (1, 32, 64, 128, 256)
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], 3, padding=1)
            for i in range(4)
        )
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256 * 7 * 7, 2048)
        self.dropout = nn.Dropout(0.8)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i % 2 == 1:
                x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)
