"""Exercise the exact homework workloads without asserting random outcomes."""

import numpy as np
import pytest

from associative_memory.models import ModelDelta, ModelHebbian, ModelZero


def pairs(rng, p=32, d=64, m=32):
    return rng.choice([-1.0, 1.0], (p, d)) / np.sqrt(d), rng.choice([-1.0, 1.0], (p, m))


def write(model, k, v):
    model.feed_data({"k": k, "v": v})
    model.optimize_parameters()


def score(model, q, v):
    before = model.state_dict()
    model.feed_data({"q": q})
    model.test()
    result = model.current_results()
    np.testing.assert_array_equal(model.W, before["W"])
    assert result["z"].shape == v.shape
    assert np.isfinite(result["z"]).all()
    return float(np.mean(result["v_hat"] == v)), float(np.mean((result["z"] - v) ** 2))


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("p", [8, 16, 32, 64, 128])
def test_capacity_workload(d, p):
    k, v = pairs(np.random.default_rng(42), p, d)
    for cls in (ModelDelta, ModelHebbian, ModelZero):
        model = cls(d=d)
        for ki, vi in zip(k, v):
            write(model, ki, vi)
        accuracy, mse = score(model, k, v)
        assert 0 <= accuracy <= 1 and mse >= 0
        if cls is ModelZero:
            assert mse == 1


@pytest.mark.parametrize("cls", [ModelDelta, ModelHebbian])
def test_partial_cue_workload(cls):
    rng = np.random.default_rng(42)
    k, v = pairs(rng)
    model = cls()
    for ki, vi in zip(k, v):
        write(model, ki, vi)
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
    model = ModelDelta()
    for _ in range(4):
        for i in rng.permutation(32):
            write(model, ka[i], va[i])
    snapshot = model.state_dict()
    original = snapshot["W"].copy()
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
        assert model.W.size == 32 * 64
        assert model.W.nbytes == 32 * 64 * 8


@pytest.mark.parametrize("cls", [ModelDelta, ModelHebbian, ModelZero])
def test_input_and_state_contracts(cls):
    model = cls(d=2, m=1)
    with pytest.raises(RuntimeError):
        model.optimize_parameters()
    with pytest.raises(ValueError):
        model.feed_data({"k": [1, 0], "v": [np.nan]})
    with pytest.raises(ValueError):
        model.feed_data({"q": [[1, 0, 0]]})
    with pytest.raises(ValueError):
        model.load_state_dict({"W": np.zeros((2, 1))})
    state = {"W": np.ones((1, 2))}
    model.load_state_dict(state)
    state["W"].fill(9)
    np.testing.assert_array_equal(model.W, [[1, 1]])
    model.feed_data({"q": [1, 0]})
    model.test()
    model.feed_data({"k": [1, 0], "v": [1]})
    with pytest.raises(RuntimeError):
        model.current_results()
    with pytest.raises(RuntimeError):
        model.test()
