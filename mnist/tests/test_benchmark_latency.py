import pytest
from utils.benchmark_latency import percentile


@pytest.mark.parametrize(("fraction", "expected"), [(0.90, 90), (0.99, 99)])
def test_nearest_rank_percentile(fraction, expected):
    assert percentile(list(range(100, 0, -1)), fraction) == expected
