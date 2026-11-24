"""Small independent checks for the experiment-one paper comparisons."""

import numpy as np
import torch

from associative_memory.experiment1 import preconditioned
from associative_memory.experiment1.preconditioned import (
    load_official_rule,
    official_write,
    scalar_control,
)
from associative_memory.experiment1.run import (
    METHODS,
    pair_tensors,
    run_trial,
    sample_pairs,
)


def test_paired_report_uses_homework_delta_as_baseline(monkeypatch):
    monkeypatch.setattr(preconditioned, "REPEATS", 2)
    audit = dict(
        max_abs_pdn_minus_scalar_score=0.0, scalar_bit_disagreements=0, tested_bits=0
    )
    rows = preconditioned.compare(4, 8, load_official_rule(), audit)
    assert [row["method"] for row in rows] == ["Delta", "PDN"]
    delta, pdn = rows
    trials = [run_trial(4, 8, i)[METHODS.index("Delta")] for i in range(2)]
    np.testing.assert_allclose(
        [delta[f"{metric}_mean"] for metric in preconditioned.METRICS],
        np.mean(trials, axis=0),
    )
    np.testing.assert_allclose(
        pdn["gain_vs_delta_pp"],
        100 * (pdn["bit_accuracy_mean"] - delta["bit_accuracy_mean"]),
    )
    assert audit["tested_bits"] == 2 * 8 * preconditioned.VALUE_DIM
    assert audit["scalar_bit_disagreements"] == 0


def test_official_pdn_scalar_control_and_stream_resume():
    keys, values = pair_tensors(*sample_pairs(4, 8, 0))
    keys, values = keys[None], values[None]
    rule = load_official_rule()
    memory, diagonal = official_write(rule, keys, values)
    actual = torch.bmm(keys, memory[:, 0])
    torch.testing.assert_close(
        actual, scalar_control(keys, values), atol=2e-5, rtol=1e-5
    )
    first = official_write(rule, keys[:, :4], values[:, :4])
    resumed = official_write(rule, keys[:, 4:], values[:, 4:], first)
    for full, partial in zip((memory, diagonal), resumed):
        torch.testing.assert_close(full, partial)
