import pytest
import torch

from associative_memory.models import ModelAttention


@pytest.mark.parametrize("single", [True, False])
def test_attention_readout_and_snapshot(single):
    generator = torch.Generator().manual_seed(123)
    keys = torch.randn(8, 4, generator=generator)
    values = torch.randn(8, 3, generator=generator)
    queries = torch.randn(5, 4, generator=generator)
    model = ModelAttention(d=4, m=3)
    for k, v in zip(keys, values):
        model.feed_data({"k": k, "v": v})
        model.optimize_parameters()
    q = queries[0] if single else queries
    expected = (q @ keys.T / 2).softmax(-1) @ values
    model.feed_data({"q": q})
    model.test()
    torch.testing.assert_close(model.current_results()["z"], expected)
    state = model.state_dict()
    model.feed_data({"k": keys[0], "v": values[0]})
    model.optimize_parameters()
    assert len(state["K"]) == 8 and len(model.K) == 9
    model.reset()
    model.load_state_dict(state)
    state["K"].zero_()
    torch.testing.assert_close(model.K, keys)
    model.feed_data({"q": q})
    model.test()
    torch.testing.assert_close(model.current_results()["z"], expected)
    before = model.state_dict()
    with pytest.raises(ValueError):
        model.load_state_dict({"K": keys, "V": values[:2]})
    torch.testing.assert_close(model.K, before["K"])
    torch.testing.assert_close(model.V, before["V"])


def test_attention_empty_and_ties():
    model = ModelAttention(d=2, m=1)
    model.feed_data({"q": torch.zeros(2)})
    with pytest.raises(RuntimeError, match="at least one"):
        model.test()
    for v in [-1.0, 1.0]:
        model.feed_data({"k": torch.tensor([1.0, 0.0]), "v": torch.tensor([v])})
        model.optimize_parameters()
    model.feed_data({"q": torch.ones(2)})
    model.test()
    torch.testing.assert_close(model.current_results()["z"], torch.zeros(1))
    assert model.current_results()["v_hat"].item() == 1


def test_attention_forward_supports_query_gradients():
    model = ModelAttention(d=2, m=1)
    for k, v in [
        (torch.tensor([1.0, 0.0]), torch.tensor([1.0])),
        (torch.tensor([0.0, 1.0]), torch.tensor([-1.0])),
    ]:
        model.feed_data({"k": k, "v": v})
        model.optimize_parameters()
    q = torch.tensor([0.3, 0.7], requires_grad=True)
    output = model(q)
    output.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    model.feed_data({"q": q})
    model.test()
    torch.testing.assert_close(model.current_results()["z"], output.detach())
