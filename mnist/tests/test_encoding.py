import numpy as np
import pytest
from encoding import latency_encode, phase_encode, rate_encode


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
