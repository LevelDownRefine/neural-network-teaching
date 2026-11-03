"""Shared deterministic MNIST split and loaders for comparison experiments."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent

def mnist_loaders(batch_size, seed=42, normalization=None):
    operations = [transforms.ToTensor()]
    if normalization is not None:
        operations.append(transforms.Normalize(*normalization))
    transform = transforms.Compose(operations)
    original = datasets.MNIST(ROOT / 'data', train=True, download=False, transform=transform)
    train, valid = random_split(original, [50000, 10000], generator=torch.Generator().manual_seed(seed))
    test = datasets.MNIST(ROOT / 'data', train=False, download=False, transform=transform)
    return tuple(DataLoader(dataset, batch_size=batch_size, shuffle=(i == 0), num_workers=0)
                 for i, dataset in enumerate((train, valid, test)))
