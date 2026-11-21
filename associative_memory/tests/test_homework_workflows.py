"""Exercise the exact homework workloads without asserting random outcomes."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from associative_memory.losses import delta_loss, hebbian_loss
from associative_memory.models import ModelMemory
from associative_memory.train import Trainer, train_epoch


def pairs(rng, p=32, d=64, m=32):
    return (
        rng.choice([-1.0, 1.0], (p, d)) / np.sqrt(d),
        rng.choice([-1.0, 1.0], (p, m)),
    )


def write(model, k, v, criterion=delta_loss):
    if criterion is None:
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    Trainer(model, criterion, optimizer).step(
        {"q": torch.as_tensor(k, dtype=torch.float32)},
        torch.as_tensor(v, dtype=torch.float32),
    )


@torch.no_grad()
def score(model, q, v):
    before = deepcopy(model.state_dict())
    z = model(q=torch.as_tensor(q, dtype=torch.float32))
    result = {
        key: value.numpy()
        for key, value in {"z": z, "v_hat": torch.where(z >= 0, 1, -1)}.items()
    }
    np.testing.assert_array_equal(model.W, before["W"])
    assert result["z"].shape == v.shape
    assert np.isfinite(result["z"]).all()
    return (
        float(np.mean(result["v_hat"] == v)),
        float(np.mean((result["z"] - v) ** 2)),
    )


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("p", [8, 16, 32, 64, 128])
def test_capacity_workload(d, p):
    k, v = pairs(np.random.default_rng(42), p, d)
    for criterion in (delta_loss, hebbian_loss, None):
        model = ModelMemory(d=d)
        if criterion is not None:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
            train_epoch(
                Trainer(model, criterion, optimizer),
                torch.tensor(k, dtype=torch.float32),
                torch.tensor(v, dtype=torch.float32),
                range(p),
            )
        accuracy, mse = score(model, k, v)
        assert 0 <= accuracy <= 1 and mse >= 0
        if criterion is None:
            assert mse == 1


@pytest.mark.parametrize("criterion", [delta_loss, hebbian_loss])
def test_partial_cue_workload(criterion):
    rng = np.random.default_rng(42)
    k, v = pairs(rng)
    model = ModelMemory()
    for ki, vi in zip(k, v):
        write(model, ki, vi, criterion)
    for p in (0, 0.25, 0.5, 0.75):
        queries = np.repeat(k, 10, axis=0)
        for q in queries:
            keep = rng.choice(64, round((1 - p) * 64), replace=False)
            mask = np.zeros(64)
            mask[keep] = 1
            q *= mask
            q /= np.linalg.norm(q)
        np.testing.assert_allclose(np.linalg.norm(queries, axis=1), 1)
        assert np.all(np.count_nonzero(queries, axis=1) == round((1 - p) * 64))
        score(model, queries, np.repeat(v, 10, axis=0))


def test_replay_workload():
    rng = np.random.default_rng(42)
    ka, va = pairs(rng)
    kb, vb = pairs(rng)
    model = ModelMemory()
    for _ in range(4):
        for i in rng.permutation(32):
            write(model, ka[i], va[i])
    snapshot = deepcopy(model.state_dict())
    original = snapshot["W"].clone()
    orders = [rng.permutation(32) for _ in range(4)]
    for condition in ("plain", "matched", 4, 16, 32):
        model.load_state_dict(snapshot)
        pool = (
            rng.choice(32, condition, replace=False)
            if isinstance(condition, int)
            else []
        )
        updates = 0
        for epoch in range(5):
            score(model, ka, va)
            score(model, kb, vb)
            if len(pool):
                score(model, ka[pool], va[pool])
                outside = np.setdiff1d(np.arange(32), pool)
                if len(outside):
                    score(model, ka[outside], va[outside])
            if epoch == 4:
                break
            for i in orders[epoch]:
                write(model, kb[i], vb[i])
                updates += 1
                if condition == "matched":
                    write(model, kb[i], vb[i])
                    updates += 1
                elif len(pool):
                    j = rng.choice(pool)
                    write(model, ka[j], va[j])
                    updates += 1
        assert updates == (128 if condition == "plain" else 256)
        np.testing.assert_array_equal(snapshot["W"], original)
        assert model.W.numel() == 32 * 64
        assert model.W.nbytes == 32 * 64 * 4


@pytest.mark.parametrize("criterion", [delta_loss, hebbian_loss])
def test_train_epoch_order_and_optimizer_state(criterion):
    keys = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    values = torch.tensor([[1.0], [-1.0]])
    model = ModelMemory(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.5)
    trainer = Trainer(model, criterion, optimizer)
    expected = torch.zeros(1, 2)
    velocity = torch.zeros_like(expected)
    for order in ([1, 0, 1], [0, 1]):
        model.eval()
        train_epoch(trainer, keys, values, order)
        for i in order:
            residual = (
                expected @ keys[i] - values[i]
                if criterion is delta_loss
                else -values[i]
            )
            velocity = 0.5 * velocity + torch.outer(residual, keys[i])
            expected -= 0.2 * velocity
        assert model.training
        torch.testing.assert_close(model.W, expected)
