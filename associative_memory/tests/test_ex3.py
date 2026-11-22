"""Check paired controls and replay accounting, independent of random scores."""

import numpy as np

from associative_memory.ex3.run import run_trial


def test_replay_controls_and_accounting():
    rows, costs = run_trial(1, np.random.SeedSequence(42))
    assert len(rows) == 25
    initial = [r for r in rows if r["b_epoch"] == 0]
    assert len({(r["a_accuracy"], r["b_accuracy"]) for r in initial}) == 1
    for cost in costs:
        expected = 128 if cost["condition"] == "sequential" else 256
        assert cost["b_updates"] == expected
        assert cost["total_updates"] == 128 + expected
        assert cost["buffer_bytes"] == cost["capacity"] * 96 * 4
        assert cost["total_bytes"] == 8192 + cost["buffer_bytes"]
    for row in rows:
        if row["capacity"]:
            inside = row["a_in_pool_accuracy"]
            outside = row["a_outside_pool_accuracy"]
            if outside is None:
                assert row["a_accuracy"] == inside
            else:
                expected = (
                    inside * row["capacity"] + outside * (32 - row["capacity"])
                ) / 32
                assert abs(row["a_accuracy"] - expected) < 1e-7
