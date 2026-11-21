"""Regression against the actual NumPy implementation from commit 960150d.

The reference source is unchanged. Only its unused base-class import is
substituted when loading it, to keep this fixture independent of current APIs.
"""

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from associative_memory.data import generate_pairs
from associative_memory.losses import delta_loss, hebbian_loss
from associative_memory.models import ModelMemory
from associative_memory.train import train_epoch


def legacy_class(rule):
    path = Path(__file__).parent / "reference/numpy_memory.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    namespace = {"ModelBase": object}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["Model" + rule]


class Comparison:
    def __init__(self, rule, d=64, m=32):
        self.old = legacy_class(rule)(d=d, m=m, eta=0.5)
        self.new = ModelMemory(d, m)
        self.optimizer = torch.optim.SGD(self.new.parameters(), lr=0.5)
        self.criterion = {"Delta": delta_loss, "Hebbian": hebbian_loss, "Zero": None}[
            rule
        ]
        self.weight_error = self.output_error = self.mse_error = 0.0
        self.flips = self.bits = 0
        self.accuracy_error = 0.0

    def train(self, keys, values, order):
        for i in order:
            self.old.feed_data({"k": keys[i], "v": values[i]})
            self.old.optimize_parameters()
            if self.criterion is not None:
                train_epoch(
                    self.new,
                    self.optimizer,
                    self.criterion,
                    torch.tensor(keys[i : i + 1], dtype=torch.float32),
                    torch.tensor(values[i : i + 1], dtype=torch.float32),
                    [0],
                )
            w = self.new.W.detach().numpy()
            np.testing.assert_allclose(w, self.old.W, atol=1e-5, rtol=1e-5)
            self.weight_error = max(
                self.weight_error, float(np.max(np.abs(w - self.old.W)))
            )

    def evaluate(self, queries, targets):
        self.old.feed_data({"q": queries})
        self.old.test()
        old = self.old.current_results()["z"]
        with torch.no_grad():
            new = self.new(q=torch.tensor(queries, dtype=torch.float32)).numpy()
        np.testing.assert_allclose(new, old, atol=2e-5, rtol=1e-5)
        self.output_error = max(self.output_error, float(np.max(np.abs(new - old))))
        self.mse_error = max(
            self.mse_error,
            abs(float(np.mean((new - targets) ** 2) - np.mean((old - targets) ** 2))),
        )
        flips = (new >= 0) != (old >= 0)
        # FP32 can flip the sign of a numerical zero: report, never hide it.
        assert np.all(np.abs(old[flips]) < 2e-5)
        assert np.all(np.abs(new[flips]) < 2e-5)
        self.accuracy_error = max(
            self.accuracy_error,
            abs(
                float(
                    np.mean(np.where(new >= 0, 1, -1) == targets)
                    - np.mean(np.where(old >= 0, 1, -1) == targets)
                )
            ),
        )
        self.flips += int(flips.sum())
        self.bits += old.size

    def report(self, label):
        print(
            f"{label}: W={self.weight_error:.3g} z={self.output_error:.3g} MSE={self.mse_error:.3g} sign_flips={self.flips}/{self.bits} accuracy_diff_pp={100 * self.accuracy_error:.6g}"
        )


@pytest.mark.parametrize("rule", ["Delta", "Hebbian", "Zero"])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("p", [8, 16, 32, 64, 128])
def test_numpy_capacity(rule, d, p):
    keys, values, order = generate_pairs(np.random.default_rng(123), p, d, 32)
    comparison = Comparison(rule, d)
    comparison.train(keys, values, order)
    comparison.evaluate(keys, values)
    comparison.report(f"capacity {rule} d={d} P={p}")


@pytest.mark.parametrize("rule", ["Delta", "Hebbian"])
def test_numpy_partial_cues(rule):
    rng = np.random.default_rng(123)
    keys, values, order = generate_pairs(rng, 32, 64, 32)
    comparison = Comparison(rule)
    comparison.train(keys, values, order)
    for fraction in (0, 0.25, 0.5, 0.75):
        queries = np.repeat(keys, 10, axis=0)
        for q in queries:
            q[rng.choice(64, int(64 * fraction), replace=False)] = 0
            q /= np.linalg.norm(q)
        comparison.evaluate(queries, np.repeat(values, 10, axis=0))
    comparison.report(f"masked {rule}")


@pytest.mark.parametrize("rule", ["Delta", "Hebbian"])
def test_numpy_replay(rule):
    rng = np.random.default_rng(123)
    ka, va, _ = generate_pairs(rng, 32, 64, 32)
    kb, vb, _ = generate_pairs(rng, 32, 64, 32)
    comparison = Comparison(rule)
    for _ in range(4):
        comparison.train(ka, va, rng.permutation(32))
    old_state = comparison.old.state_dict()
    new_state = deepcopy(comparison.new.state_dict())
    orders = [rng.permutation(32) for _ in range(4)]
    for condition in ("plain", "matched", 4, 16, 32):
        comparison.old.load_state_dict(old_state)
        comparison.new.load_state_dict(new_state)
        pool = (
            rng.choice(32, condition, replace=False)
            if isinstance(condition, int)
            else []
        )
        for epoch in range(5):
            comparison.evaluate(ka, va)
            comparison.evaluate(kb, vb)
            if len(pool):
                comparison.evaluate(ka[pool], va[pool])
                outside = np.setdiff1d(np.arange(32), pool)
                if len(outside):
                    comparison.evaluate(ka[outside], va[outside])
            if epoch == 4:
                break
            order = []
            for i in orders[epoch]:
                order.append(32 + i)
                if condition == "matched":
                    order.append(32 + i)
                elif len(pool):
                    order.append(rng.choice(pool))
            comparison.train(np.concatenate([ka, kb]), np.concatenate([va, vb]), order)
    comparison.report(f"replay {rule}")
