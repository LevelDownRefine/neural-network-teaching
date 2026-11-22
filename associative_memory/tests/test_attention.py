"""Compare explicit attention with SGD-based Hebbian writes after every update."""

import pytest
import torch

from associative_memory.models import ModelAttention, ModelMemory
from associative_memory.train import AttentionTrainer, HebbianTrainer, train_epoch


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


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.Adam])
def test_projected_attention_and_differentiable_hebbian_train_identically(
    optimizer_class,
):
    from copy import deepcopy

    torch.manual_seed(123)
    projections = torch.nn.ModuleDict(
        {
            "q": torch.nn.Linear(3, 4, bias=False),
            "k": torch.nn.Linear(3, 4, bias=False),
            "v": torch.nn.Linear(3, 2, bias=False),
        }
    )
    hebbian_projections = deepcopy(projections)
    optimizers = [
        optimizer_class(net.parameters(), lr=0.01)
        for net in (projections, hebbian_projections)
    ]
    queries, context = torch.randn(5, 3), torch.randn(7, 3)
    targets = torch.randn(5, 2)
    attention = ModelAttention(eta=0.5)
    initial = [p.detach().clone() for p in projections.parameters()]
    first_loss = None
    max_output_error = max_gradient_error = max_parameter_error = 0.0
    for _ in range(30):
        outputs, losses = [], []
        for net, optimizer, compressed in zip(
            (projections, hebbian_projections), optimizers, (False, True)
        ):
            optimizer.zero_grad()
            q, k, v = net["q"](queries), net["k"](context), net["v"](context)
            if compressed:
                # Rebuild fast memory each forward. No detach or optimizer on M:
                # all outer-product writes remain differentiable w.r.t. K and V.
                memory = k.new_zeros(v.shape[-1], k.shape[-1])
                for key, value in zip(k, v):
                    memory = memory + attention.eta * torch.outer(value, key)
                output = q @ memory.T
            else:
                output = attention(q=q, k=k, v=v)
            loss = (output - targets).square().mean()
            loss.backward()
            outputs.append(output.detach())
            losses.append(loss.detach())
        if first_loss is None:
            first_loss = losses[0].item()
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-5)
        max_output_error = max(
            max_output_error, (outputs[0] - outputs[1]).abs().max().item()
        )
        for a, b in zip(projections.parameters(), hebbian_projections.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-5)
            max_gradient_error = max(
                max_gradient_error, (a.grad - b.grad).abs().max().item()
            )
        for optimizer in optimizers:
            optimizer.step()
        for a, b in zip(projections.parameters(), hebbian_projections.parameters()):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
            max_parameter_error = max(max_parameter_error, (a - b).abs().max().item())
    assert losses[0].item() < first_loss
    assert any(not torch.equal(a, b) for a, b in zip(initial, projections.parameters()))
    print(
        f"{optimizer_class.__name__}: loss {first_loss:.6g} -> {losses[0].item():.6g}; "
        f"max errors output={max_output_error:.3g}, gradient={max_gradient_error:.3g}, "
        f"parameter={max_parameter_error:.3g}"
    )


@pytest.mark.parametrize("eta", [0.25, 0.5, 1.0])
def test_attention_trainer_matches_hebbian_weights_losses_and_outputs(eta):
    rng = torch.Generator().manual_seed(123)
    k = torch.randn(12, 4, generator=rng)
    v = torch.randn(12, 3, generator=rng)
    q = torch.randn(5, 4, generator=rng)
    memory = ModelMemory(4, 3)
    hebbian = HebbianTrainer(memory, lr=eta)
    attention = ModelAttention(eta=eta)
    writer = AttentionTrainer(attention, d=4, m=3)
    torch.testing.assert_close(writer.memory_weight(), memory.W)
    order = torch.cat([torch.randperm(12, generator=rng), torch.tensor([3, 3, 0])])
    for i in order:
        a = hebbian.step({"q": k[i]}, v[i])
        b = writer.step({"q": k[i]}, v[i])
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            writer.memory_weight(), memory.W, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            attention(q=q, k=writer.keys, v=writer.values),
            memory(q=q),
            atol=2e-5,
            rtol=2e-5,
        )
    train_epoch(writer, k, v, order)
    train_epoch(hebbian, k, v, order)
    torch.testing.assert_close(writer.memory_weight(), memory.W, atol=2e-6, rtol=2e-5)
    restored = ModelMemory(4, 3)
    restored.load_state_dict({"W": writer.memory_weight()})
    torch.testing.assert_close(restored(q=q), memory(q=q), atol=2e-5, rtol=2e-5)
    assert writer.optimizer is None
    assert not writer.keys.requires_grad and not writer.values.requires_grad
    before = writer.memory_weight().clone()
    with pytest.raises(ValueError):
        writer.step({"q": k[0]}, torch.tensor([float("nan"), 0.0, 0.0]))
    torch.testing.assert_close(writer.memory_weight(), before)
    k.zero_()
    v.zero_()
    torch.testing.assert_close(writer.memory_weight(), before)
