import torch
from data_pipeline import mnist as common_data
from torchvision import transforms


class FakeMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train, download, transform):
        self.length = 60000 if train else 10000
        self.transform = transform
        assert download is False

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index, index % 10


def test_split_is_deterministic_and_no_leakage(monkeypatch):
    monkeypatch.setattr(common_data.datasets, "MNIST", FakeMNIST)
    train, validation, test = common_data.mnist_loaders(64, 42)
    train_again, validation_again, _ = common_data.mnist_loaders(64, 42)
    assert len(train.dataset) == 50000
    assert len(validation.dataset) == len(test.dataset) == 10000
    assert train.dataset.indices == train_again.dataset.indices
    assert validation.dataset.indices == validation_again.dataset.indices
    assert set(train.dataset.indices).isdisjoint(validation.dataset.indices)


def test_augmentation_is_training_only(monkeypatch):
    monkeypatch.setattr(common_data.datasets, "MNIST", FakeMNIST)
    train, validation, test = common_data.mnist_loaders(64, 42, augmentation=True)
    assert any(
        isinstance(t, transforms.RandomRotation)
        for t in train.dataset.dataset.transform.transforms
    )
    assert not any(
        isinstance(t, transforms.RandomRotation)
        for t in validation.dataset.dataset.transform.transforms
    )
    assert not any(
        isinstance(t, transforms.RandomRotation)
        for t in test.dataset.transform.transforms
    )
