"""Training-only transformations for the MNIST M5 experiment."""

from torchvision import transforms


def m5_augmentation():
    return [
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
    ]
