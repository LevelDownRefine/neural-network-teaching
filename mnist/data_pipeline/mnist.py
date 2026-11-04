"""Deterministic splits, with augmentation confined to training samples."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data_pipeline.augmentation import m5_augmentation

ROOT = Path(__file__).resolve().parent.parent


def mnist_loaders(batch_size, seed=42, normalization=None, augmentation=False):
    base = [transforms.ToTensor()]
    if normalization is not None:
        base.append(transforms.Normalize(*normalization))
    train_transform = transforms.Compose(
        (m5_augmentation() if augmentation else []) + base
    )
    evaluation_transform = transforms.Compose(base)
    original = datasets.MNIST(
        ROOT / "data", train=True, download=False, transform=train_transform
    )
    validation_data = datasets.MNIST(
        ROOT / "data", train=True, download=False, transform=evaluation_transform
    )
    indices = torch.randperm(
        len(original), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train = Subset(original, indices[:50000])
    valid = Subset(validation_data, indices[50000:])
    test = datasets.MNIST(
        ROOT / "data", train=False, download=False, transform=evaluation_transform
    )
    return tuple(
        DataLoader(dataset, batch_size=batch_size, shuffle=(i == 0), num_workers=0)
        for i, dataset in enumerate((train, valid, test))
    )
