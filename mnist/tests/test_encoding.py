from pathlib import Path

import pytest
import torch
import yaml
from encoding import encode_input


def test_repeat_preserves_values_shape_and_gradients():
    images = torch.tensor([[[[-0.5, 1.5], [2.0, 0.0]]]], requires_grad=True)
    encoded = encode_input(images, 3)
    assert encoded.shape == (1, 1, 2, 2, 3)
    assert encoded.dtype == images.dtype and encoded.device == images.device
    for step in range(3):
        torch.testing.assert_close(encoded[..., step], images)
    encoded.sum().backward()
    torch.testing.assert_close(images.grad, torch.full_like(images, 3))


@pytest.mark.parametrize("method", ["repeat", "rate", "latency", "phase", "rank_order"])
def test_encoders_share_batch_shape_and_dtype(method):
    images = torch.tensor([[[[0.0, 0.5, 1.0]]], [[[1.0, 0.5, 0.0]]]])
    kwargs = {"period": 5} if method == "phase" else {}
    encoded = encode_input(images, 20, method, **kwargs)
    assert encoded.shape == (2, 1, 1, 3, 20)
    assert encoded.dtype == images.dtype and encoded.device == images.device
    if method != "repeat":
        assert ((encoded == 0) | (encoded == 1)).all()


def test_spike_encodings_preserve_background_and_intensity_order():
    images = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    latency = encode_input(images, 10, "latency")[0, 0, 0]
    assert latency[0].sum() == 0
    assert latency[1].sum() == latency[2].sum() == 1
    assert latency[2].argmax() < latency[1].argmax()
    phase = encode_input(images, 20, "phase", period=5)[0, 0, 0]
    assert phase[0].sum() == 0
    assert phase[1].sum() == phase[2].sum() == 4
    assert phase[2].nonzero().flatten().tolist() == [0, 5, 10, 15]
    rate = encode_input(
        images, 10_000, "rate", generator=torch.Generator().manual_seed(42)
    )
    again = encode_input(
        images, 10_000, "rate", generator=torch.Generator().manual_seed(42)
    )
    assert torch.equal(rate, again)
    assert rate[0, 0, 0, 0].sum() == 0
    assert rate[0, 0, 0, 2].sum() == 10_000
    assert rate[0, 0, 0, 1].mean().item() == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    "method", ["rate", "latency", "phase", "rank_order", "population"]
)
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_spike_encodings_reject_invalid_intensity(method, value):
    with pytest.raises(ValueError, match="intensities"):
        encode_input(torch.tensor([[[[value]]]]), 10, method, period=5)


def test_encoding_rejects_invalid_arguments():
    images = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="shape"):
        encode_input(torch.zeros(2, 2), 5)
    for steps in (0, -1, True):
        with pytest.raises(ValueError, match="steps"):
            encode_input(images, steps)
    with pytest.raises(ValueError, match="unknown encoding"):
        encode_input(images, 10, "unknown")
    for period in (None, 0, 11, True):
        with pytest.raises(ValueError, match="period"):
            encode_input(images, 10, "phase", period=period)
    with pytest.raises(ValueError, match="floating-point"):
        encode_input(images.to(torch.int64), 10, "rate")
    for size in (0, 1, True):
        with pytest.raises(ValueError, match="population_size"):
            encode_input(images, 10, "population", population_size=size)
    for width in (0, -1, float("nan"), True):
        with pytest.raises(ValueError, match="population_width"):
            encode_input(images, 10, "population", population_width=width)


def test_rank_order_preserves_ties_and_depends_on_rank_not_magnitude():
    images = torch.tensor([[[[0.0, 0.2, 0.6, 1.0]]], [[[0.0, 0.1, 0.2, 0.3]]]])
    encoded = encode_input(images, 12, "rank_order")
    torch.testing.assert_close(encoded[0], encoded[1])
    assert encoded[:, :, :, 0].sum() == 0
    assert encoded[0, 0, 0, 3].argmax() == 0
    assert encoded[0, 0, 0, 2].argmax() == 4
    assert encoded[0, 0, 0, 1].argmax() == 8
    ties = encode_input(torch.tensor([[[[0.0, 0.5, 0.5, 1.0]]]]), 8, "rank_order")
    torch.testing.assert_close(ties[0, 0, 0, 1], ties[0, 0, 0, 2])
    assert encode_input(torch.zeros(1, 1, 2, 2), 8, "rank_order").sum() == 0


def test_population_coding_adds_channels_and_responds_to_preferred_value():
    images = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    kwargs = {"method": "population", "population_size": 3}
    encoded = encode_input(
        images, 100, generator=torch.Generator().manual_seed(42), **kwargs
    )
    again = encode_input(
        images, 100, generator=torch.Generator().manual_seed(42), **kwargs
    )
    assert encoded.shape == (1, 3, 1, 3, 100)
    assert encoded.dtype == images.dtype and encoded.device == images.device
    assert ((encoded == 0) | (encoded == 1)).all()
    assert torch.equal(encoded, again)
    assert encoded[:, :, :, 0].sum() == 0
    assert encoded[0, 1, 0, 1].sum() == 100
    assert encoded[0, 2, 0, 2].sum() == 100
    multi_channel = torch.tensor([[[[0.0]], [[1.0]]]])
    result = encode_input(multi_channel, 10, "population")
    assert result.shape == (1, 6, 1, 1, 10)
    assert result[0, :3].sum() == 0
    assert result[0, 5].sum() == 10


def test_repeat_supports_fullgraph_compilation():
    images = torch.randn(2, 1, 4, 4)
    compiled = torch.compile(encode_input, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(images, 5), encode_input(images, 5))


def test_single_channel_encodings_feed_existing_tssl_model(monkeypatch):
    pytest.importorskip("snntorch")
    tssl_dir = Path(__file__).resolve().parents[1] / "models" / "tssl"
    monkeypatch.syspath_prepend(str(tssl_dir))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *args, **kwargs: self)
    import cnns

    config = yaml.safe_load((tssl_dir / "Networks" / "MNIST_CNN.yaml").read_text())
    network = cnns.Network(config["Network"], config["Layers"], [1, 28, 28])
    network.eval()
    images = torch.rand(2, 1, 28, 28)
    steps = config["Network"]["n_steps"]
    with torch.no_grad():
        population = encode_input(images, steps, "population")
        assert population.shape[1] == 3
        assert config["Layers"]["conv_1"]["in_channels"] == 1
        baseline = network.forward(
            images.unsqueeze(-1).repeat(1, 1, 1, 1, steps), 0, False
        )
        for method in ("repeat", "rate", "latency", "phase", "rank_order"):
            kwargs = {"period": steps} if method == "phase" else {}
            output = network.forward(
                encode_input(images, steps, method, **kwargs), 0, False
            )
            assert output.shape == baseline.shape
            assert torch.isfinite(output).all()
            if method == "repeat":
                torch.testing.assert_close(output, baseline, rtol=0, atol=0)
