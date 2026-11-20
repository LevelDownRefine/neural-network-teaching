import numpy as np
import pytest

from associative_memory.random import make_rng


def test_reproducible_and_independent_streams():
    expected = make_rng(42, "experiment1", "order", 64, 32).permutation(32)
    make_rng(42, "experiment1", "data", 64, 32).random(10000)
    np.testing.assert_array_equal(
        expected, make_rng(42, "experiment1", "order", 64, 32).permutation(32)
    )
    for args in [
        (43, "experiment1", "order", 64, 32),
        (42, "experiment2", "order", 64, 32),
        (42, "experiment1", "masks", 64, 32),
        (42, "experiment1", "order", 32, 64),
    ]:
        assert not np.array_equal(expected, make_rng(*args).permutation(32))


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        make_rng(seed, "experiment1", "data")
