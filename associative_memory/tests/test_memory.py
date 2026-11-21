"""Compare with the unmodified FLA reference and independent closed forms."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from associative_memory.models import ModelDelta, ModelHebbian, ModelZero


def fla_reference():
    # Extract only the reference function, avoiding FLA/Triton/einops imports.
    path = Path(__file__).parent / "reference" / "fla_gated_delta.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "naive_recurrent_gated_delta_rule"
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
    model = ModelDelta(d=d, m=m, eta=eta)
    model.load_state_dict({"W": torch.tensor(initial, dtype=torch.float32)})
    predictions = []
    for k, v, q in zip(keys, values, queries):
        model.feed_data({"k": torch.as_tensor(k), "v": torch.as_tensor(v)})
        model.optimize_parameters()
        model.feed_data({"q": torch.as_tensor(q)})
        model.test()
        predictions.append(model.current_results()["z"])
    np.testing.assert_allclose(
        predictions, output.numpy()[0, :, 0], atol=2e-5, rtol=2e-5
    )
    np.testing.assert_allclose(model.W, state.numpy()[0, 0].T, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("eta", [0.25, 0.5, 1.0])
def test_hebbian_closed_form(eta):
    rng = np.random.default_rng(7)
    keys = rng.choice([-1.0, 1.0], (32, 64)) / 8
    values = rng.choice([-1.0, 1.0], (32, 32))
    model = ModelHebbian(eta=eta)
    for k, v in zip(keys, values):
        model.feed_data({"k": torch.as_tensor(k), "v": torch.as_tensor(v)})
        model.optimize_parameters()
    np.testing.assert_allclose(model.W, eta * values.T @ keys, atol=1e-14)
    model.feed_data({"q": torch.as_tensor(keys)})
    model.test()
    np.testing.assert_allclose(
        model.current_results()["z"], eta * (keys @ keys.T) @ values, atol=1e-14
    )


@pytest.mark.parametrize("cls", [ModelDelta, ModelHebbian, ModelZero])
def test_snapshot_and_reset(cls):
    model = cls(d=2, m=3, eta=0.25)
    model.feed_data(
        {"k": torch.as_tensor([1.0, 0.0]), "v": torch.as_tensor([-1.0, 1.0, -1.0])}
    )
    model.optimize_parameters()
    state = model.state_dict()
    assert not np.shares_memory(state["W"], model.W)
    model.reset()
    np.testing.assert_array_equal(model.W, np.zeros((3, 2)))
    with pytest.raises(RuntimeError):
        model.current_results()
    model.load_state_dict(state=state)
    np.testing.assert_array_equal(model.W, state["W"])
    assert model.eta == 0.25
    before = model.W.clone()
    model.feed_data({"q": torch.as_tensor([[1.0, 0.0], [0.0, 1.0]])})
    model.test()
    result = model.current_results()
    np.testing.assert_array_equal(model.W, before)
    np.testing.assert_array_equal(result["v_hat"][1], [1, 1, 1])
    if cls is ModelZero:
        np.testing.assert_array_equal(result["z"], np.zeros((2, 3)))
    else:
        np.testing.assert_array_equal(result["v_hat"][0], [-1, 1, -1])


def test_hebbian_attention_readout():
    from torch.nn.functional import scaled_dot_product_attention

    generator = torch.Generator().manual_seed(123)
    k, v, q = [
        torch.randn(shape, generator=generator, dtype=torch.float32)
        for shape in [(7, 4), (7, 3), (5, 4)]
    ]
    model = ModelHebbian(d=4, m=3)
    for key, value in zip(k, v):
        model.feed_data({"k": key, "v": value})
        model.optimize_parameters()
    model.feed_data({"q": q})
    model.test()
    scores = q @ k.T
    torch.testing.assert_close(
        torch.from_numpy(model.current_results()["z"]), model.eta * scores @ v
    )
    torch.testing.assert_close(
        (scores / 2).softmax(-1) @ v,
        scaled_dot_product_attention(q[None], k[None], v[None])[0],
    )
