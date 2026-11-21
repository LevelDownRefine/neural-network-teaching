import numpy as np

from associative_memory.data import generate_pairs


def test_pairs_and_order():
    first = generate_pairs(np.random.default_rng(42), 32, 64, 16)
    second = generate_pairs(np.random.default_rng(42), 32, 64, 16)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    k, v, order = first
    assert k.shape == (32, 64) and v.shape == (32, 16)
    assert set(np.unique(k)) == {-1 / 8, 1 / 8}
    assert set(np.unique(v)) == {-1, 1}
    np.testing.assert_allclose(np.linalg.norm(k, axis=1), 1)
    np.testing.assert_array_equal(np.sort(order), np.arange(32))
    rng = np.random.default_rng(42)
    generate_pairs(rng, 32, 64, 16)
    assert not np.array_equal(generate_pairs(rng, 32, 64, 16)[0], k)
