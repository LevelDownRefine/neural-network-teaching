"""Small independent checks for the experiment-one paper comparisons."""

from itertools import repeat

import numpy as np
import torch

from associative_memory.experiment1.optimized_writes import (
    RIDGE,
    evaluate,
    fit_delta,
    fit_rls,
)
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


def test_comparators_match_baseline_and_batch_ridge():
    keys, values = pair_tensors(*sample_pairs(4, 8, 0))
    metrics, _ = evaluate(fit_delta(keys, values, repeat(0.5, 8)), keys, values.numpy())
    np.testing.assert_allclose(metrics, run_trial(4, 8, 0)[METHODS.index("Delta")])
    model = fit_rls(keys, values)
    x = keys.double()
    expected = torch.linalg.solve(x.T @ x + RIDGE * torch.eye(4), x.T @ values.double())
    torch.testing.assert_close(model.W.double().T, expected, atol=4e-4, rtol=2e-4)


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
