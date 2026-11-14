import numpy as np
import pytest
import torch
from encoding import latency_encode, phase_encode, rate_encode, repeat_static_input


def test_repeat_static_input_keeps_values_shape_and_gradients():
    images = torch.tensor([[[[-0.5, 1.5], [2.0, 0.0]]]], requires_grad=True)
    encoded = repeat_static_input(images, 3)
    assert encoded.shape == (1, 1, 2, 2, 3)
    assert encoded.dtype == images.dtype and encoded.device == images.device
    for step in range(3):
        torch.testing.assert_close(encoded[..., step], images)
    encoded.sum().backward()
    torch.testing.assert_close(images.grad, torch.full_like(images, 3))


def test_repeat_static_input_rejects_invalid_shape_and_steps():
    with pytest.raises(ValueError, match="shape"):
        repeat_static_input(torch.zeros(28, 28), 5)
    for steps in (0, -1, True):
        with pytest.raises(ValueError, match="steps"):
            repeat_static_input(torch.zeros(1, 1, 2, 2), steps)


def test_repeat_static_input_supports_fullgraph_compilation():
    images = torch.randn(2, 1, 4, 4)
    compiled = torch.compile(repeat_static_input, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(images, 5), repeat_static_input(images, 5))


def test_encodings_preserve_background_and_intensity_order():
    image = np.array([[0, 0.5, 1]], dtype=np.float32)
    latency = latency_encode(image, 10)
    assert latency[:, 0, 0].sum() == 0
    assert latency[:, 0, 1].sum() == latency[:, 0, 2].sum() == 1
    assert latency[:, 0, 2].argmax() < latency[:, 0, 1].argmax()
    phase = phase_encode(image, 20, 5)
    assert phase[:, 0, 0].sum() == 0
    assert phase[:, 0, 1].sum() == phase[:, 0, 2].sum() == 4
    np.testing.assert_array_equal(np.flatnonzero(phase[:, 0, 2]), [0, 5, 10, 15])
    rate = rate_encode(image, 100, np.random.default_rng(42))
    assert rate[:, 0, 0].sum() == 0
    assert rate[:, 0, 2].sum() == 100
    assert np.array_equal(rate, rate_encode(image, 100, np.random.default_rng(42)))


def test_encodings_reject_invalid_inputs():
    with pytest.raises(ValueError):
        latency_encode(np.array([[1.1]]), 10)
    with pytest.raises(ValueError):
        phase_encode(np.ones((2, 2)), 10, 11)
    with pytest.raises(ValueError):
        rate_encode(np.ones((2, 2)), 0, np.random.default_rng(0))
