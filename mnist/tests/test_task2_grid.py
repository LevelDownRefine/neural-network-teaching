import json
from types import SimpleNamespace

import pytest

from spike_encoding.run_encoding_comparison import matching_result, write_report


def test_existing_result_must_match_the_complete_experiment(tmp_path):
    args = SimpleNamespace(seed=42, epochs=100, batch_size=20, phase_period=20)
    result = {
        "config": {
            "encoding": "rate",
            "steps": 10,
            "seed": 42,
            "epochs": 100,
            "batch_size": 20,
            "augment": True,
            "ema": True,
            "max_batches": 0,
            "phase_period": 20,
        },
        "test": {"count": 10_000},
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert matching_result(path, args, "rate", 10) == result
    with pytest.raises(ValueError, match="configuration"):
        matching_result(path, args, "latency", 10)
    result["test"]["count"] = 20
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="full test-set"):
        matching_result(path, args, "rate", 10)


def test_partial_report_contains_metrics_and_plots(tmp_path):
    rows = [
        {
            "encoding": "rate",
            "steps": 10,
            "accuracy": 0.9,
            "input_spikes_per_image": 50,
            "hidden_spikes_per_image": 20,
            "model_only_batch_ms": 1,
            "encoding_and_model_batch_ms": 2,
            "latency_batch_size": 20,
        }
    ]
    write_report(rows, tmp_path)
    assert "rate,10,0.9" in (tmp_path / "summary.csv").read_text(encoding="utf-8-sig")
    for name in ("accuracy_vs_steps", "spikes_vs_steps", "accuracy_vs_spikes"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
