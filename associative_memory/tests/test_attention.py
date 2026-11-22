"""Compare explicit attention with SGD-based Hebbian writes after every update."""

import pytest
import torch

from associative_memory.models import ModelAttention, ModelMemory
from associative_memory.train import HebbianTrainer


@pytest.mark.parametrize("eta", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("d,m", [(4, 3), (64, 32)])
def test_attention_matches_hebbian_after_every_write(eta, d, m):
    rng = torch.Generator().manual_seed(123)
    keys = torch.randn(16, d, generator=rng) / d**0.5
    values = torch.randn(16, m, generator=rng)
    queries = torch.randn(7, d, generator=rng)
    queries[0, ::2] = 0  # Partial cue; queries need not equal stored keys.
    memory = ModelMemory(d, m)
    trainer = HebbianTrainer(memory, lr=eta)
    attention = ModelAttention(eta=eta)
    order = torch.cat([torch.randperm(16, generator=rng), torch.tensor([3, 3, 0])])
    assert not list(attention.parameters())
    torch.testing.assert_close(
        attention(q=queries, k=keys[:0], v=values[:0]), memory(q=queries)
    )
    for step, index in enumerate(order):
        trainer.step({"q": keys[index]}, values[index])
        selected = order[: step + 1]
        for q in (queries, queries[0], keys[index]):
            with torch.no_grad():
                actual = attention(q=q, k=keys[selected], v=values[selected])
                expected = memory(q=q)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_attention_propagates_gradients():
    rng = torch.Generator().manual_seed(123)
    q = torch.randn(2, 4, generator=rng, requires_grad=True)
    k = torch.randn(5, 4, generator=rng, requires_grad=True)
    v = torch.randn(5, 3, generator=rng, requires_grad=True)
    attention = ModelAttention()
    explicit = attention(q=q, k=k, v=v)
    compressed = q @ (0.5 * v.T @ k).T
    grad_explicit = torch.autograd.grad(explicit.square().sum(), (q, k, v))
    grad_compressed = torch.autograd.grad(compressed.square().sum(), (q, k, v))
    for a, b in zip(grad_explicit, grad_compressed):
        torch.testing.assert_close(a, b)


def test_attention_input_contracts():
    attention = ModelAttention()
    q, k, v = torch.ones(2), torch.ones(3, 2), torch.ones(3, 1)
    with pytest.raises(AssertionError, match="float32"):
        attention(q=q.double(), k=k, v=v)
    with pytest.raises(ValueError):
        attention(q=q, k=k, v=v[:2])
    with pytest.raises(ValueError):
        attention(q=q, k=k, v=torch.full_like(v, float("nan")))
