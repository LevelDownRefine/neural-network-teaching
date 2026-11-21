"""Compare with the unmodified FLA reference and independent closed forms."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from associative_memory.models import ModelMemory
from associative_memory.train import DeltaTrainer, HebbianTrainer


def fla_reference():
    path = Path(__file__).parent / "reference" / "fla_gated_delta.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "naive_recurrent_gated_delta_rule"
        )
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("eta", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("d,m,length", [(4, 3, 9), (64, 32, 32)])
def test_delta_matches_fla(seed, eta, d, m, length):
    rng = np.random.default_rng(seed)
    keys = rng.choice([-1.0, 1.0], (length, d)) / np.sqrt(d)
    values = rng.choice([-1.0, 1.0], (length, m))
    queries = rng.normal(size=(length, d))
    initial = rng.normal(scale=0.05, size=(m, d))

    def tensor(x):
        return torch.tensor(x, dtype=torch.float32)[None, :, None]

    output, state = fla_reference()(
        tensor(queries),
        tensor(keys),
        tensor(values),
        torch.full((1, length, 1), eta),
        torch.zeros(1, length, 1),
        scale=1.0,
        initial_state=torch.tensor(initial.T.copy())[None, None],
        output_final_state=True,
    )
    model = ModelMemory(d=d, m=m)
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    model.load_state_dict({"W": torch.tensor(initial, dtype=torch.float32)})
    predictions = []
    for k, v, q in zip(keys, values, queries):
        optimizer.zero_grad()
        loss = DeltaTrainer.loss(
            model(q=torch.tensor(k, dtype=torch.float32)),
            torch.tensor(v, dtype=torch.float32),
        )
        loss.backward()
        optimizer.step()
        z = model(q=torch.as_tensor(q, dtype=torch.float32))
        predictions.append(z.detach().numpy())
    np.testing.assert_allclose(
        predictions, output.numpy()[0, :, 0], atol=2e-05, rtol=2e-05
    )
    np.testing.assert_allclose(
        model.W.detach(), state.numpy()[0, 0].T, atol=2e-06, rtol=2e-05
    )


@pytest.mark.parametrize("criterion", [DeltaTrainer.loss, HebbianTrainer.loss])
@pytest.mark.parametrize("eta", [0.25, 0.5, 1.0])
def test_sgd_matches_every_explicit_update(criterion, eta):
    rng = torch.Generator().manual_seed(123)
    model = ModelMemory(d=7, m=5)
    initial = torch.randn(5, 7, generator=rng)
    model.load_state_dict({"W": initial})
    expected = initial.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    for _ in range(20):
        k = torch.randn(7, generator=rng) / 7**0.5
        v = torch.randn(5, generator=rng)
        error = v - expected @ k if criterion is DeltaTrainer.loss else v
        expected += eta * torch.outer(error, k)
        optimizer.zero_grad()
        criterion(model(q=k), v).backward()
        optimizer.step()
        torch.testing.assert_close(model.W, expected)


def test_hebbian_closed_form():
    rng = torch.Generator().manual_seed(123)
    k = torch.randn(12, 4, generator=rng)
    v = torch.randn(12, 3, generator=rng)
    model = ModelMemory(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for key, value in zip(k, v):
        optimizer.zero_grad()
        HebbianTrainer.loss(model(q=key), value).backward()
        optimizer.step()
    torch.testing.assert_close(model.W, 0.5 * v.T @ k)
    torch.testing.assert_close(model(q=k), 0.5 * (k @ k.T) @ v)


def test_native_state_and_module_composition():
    from copy import deepcopy

    model = ModelMemory(2, 1)
    assert dict(model.named_parameters())["W"] is model.W
    assert not dict(model.named_buffers())
    snapshot = deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    DeltaTrainer.loss(model(q=torch.ones(2)), torch.ones(1)).backward()
    optimizer.step()
    assert model.W.abs().sum() > 0
    assert snapshot["W"].count_nonzero() == 0
    model.load_state_dict(snapshot, strict=True)
    torch.testing.assert_close(model(q=torch.ones(2)), torch.zeros(1))
    container = torch.nn.Sequential(model)
    state = deepcopy(container.state_dict())
    container.load_state_dict(state)
    assert "0.W" in state


@pytest.mark.parametrize("criterion", [DeltaTrainer.loss, HebbianTrainer.loss])
def test_input_contracts_and_gradients(criterion):
    model = ModelMemory(2, 1)
    with pytest.raises(TypeError):
        model(q=np.ones(2))
    with pytest.raises(AssertionError, match="float32"):
        model(q=torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        model(q=torch.ones(3))
    with pytest.raises(ValueError):
        model(q=torch.tensor([float("nan"), 1.0]))
    with pytest.raises(ValueError):
        criterion(model(q=torch.ones(2)), torch.ones(2))
    with pytest.raises(ValueError):
        criterion(model(q=torch.ones(2)), torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        criterion(model(q=torch.ones(3, 2)), torch.ones(3, 1))
    criterion(model(q=torch.ones(2)), torch.ones(1)).backward()
    assert model.W.grad is not None
    with torch.no_grad():
        assert not model(q=torch.ones(2)).requires_grad
