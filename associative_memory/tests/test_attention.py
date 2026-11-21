import pytest
import torch

from associative_memory.models import ModelAttention


@pytest.mark.parametrize("single", [True, False])
def test_attention_readout_and_gradients(single):
    rng = torch.Generator().manual_seed(123)
    k = torch.randn(8, 4, generator=rng, requires_grad=True)
    v = torch.randn(8, 3, generator=rng, requires_grad=True)
    q = torch.randn((4,) if single else (5, 4), generator=rng, requires_grad=True)
    model = ModelAttention(4, 3)
    output = model(q=q, k=k, v=v)
    expected = (q @ k.T / 2).softmax(-1) @ v
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in (q, k, v))
    assert not list(model.parameters())
    assert not model.state_dict()


def test_attention_validates_memory():
    model = ModelAttention(2, 1)
    for k, v in [
        (torch.empty(0, 2), torch.empty(0, 1)),
        (torch.ones(3, 2), torch.ones(2, 1)),
        (torch.ones(2), torch.ones(1)),
    ]:
        with pytest.raises(ValueError):
            model(q=torch.ones(2), k=k, v=v)
    output = model(q=torch.ones(2), k=torch.ones(2, 2), v=torch.tensor([[-1.0], [1.0]]))
    torch.testing.assert_close(output, torch.zeros(1))


def test_attention_trains_upstream_projection():
    projection = torch.nn.Linear(2, 2, bias=False)
    attention = ModelAttention(2, 1)
    optimizer = torch.optim.SGD(projection.parameters(), lr=0.1)
    before = projection.weight.detach().clone()
    optimizer.zero_grad()
    output = attention(
        q=projection(torch.tensor([0.3, 0.7])),
        k=torch.eye(2),
        v=torch.tensor([[1.0], [-1.0]]),
    )
    (output - 2).square().sum().backward()
    optimizer.step()
    assert not torch.equal(before, projection.weight)


def test_shared_keyword_batch():
    from associative_memory.models import ModelMemory

    batch = {"q": torch.ones(2), "k": torch.eye(2), "v": torch.ones(2, 1)}
    memory = ModelMemory(2, 1)
    attention = ModelAttention(2, 1)
    torch.testing.assert_close(memory(**batch), torch.zeros(1))
    torch.testing.assert_close(attention(**batch), torch.ones(1))
    for model in (memory, attention):
        with pytest.raises(KeyError):
            model(k=batch["k"], v=batch["v"])
