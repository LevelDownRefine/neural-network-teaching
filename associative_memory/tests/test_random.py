import numpy as np

from associative_memory.random import make_rng


def test_reproducible_generators():
    first = make_rng(42)
    second = make_rng(42)
    np.testing.assert_array_equal(first.random(32), second.random(32))
    expected = make_rng(42).random(32)
    make_rng(7).random(100)
    np.testing.assert_array_equal(expected, make_rng(42).random(32))
    assert not np.array_equal(expected, make_rng(7).random(32))
