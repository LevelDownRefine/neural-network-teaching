import json

import numpy as np
import pytest
from utils import build_comparison
from utils.classification_metrics import calculate


def test_report_includes_available_current_experiment(tmp_path, monkeypatch):
    root = tmp_path / "mnist"
    results = root / "results"
    output = results / "comparison"
    path = results / "any_new_tag" / "metrics.json"
    path.parent.mkdir(parents=True)
    labels = np.arange(10)
    path.write_text(
        json.dumps(
            {
                "config": {"seed": 77},
                "selection": "official test accuracy (test set used for selection)",
                "test": calculate(labels, np.eye(10)),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_comparison, "ROOT", root)
    monkeypatch.setattr(build_comparison, "OUT", output)
    build_comparison.main()
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["available"][0]["model"] == "any_new_tag"
    assert report["available"][0]["test_accuracy"] == 1.0
    assert report["available"][0]["seed"] == 77
    assert "test set used for selection" in report["available"][0]["selection"]


def test_discovery_skips_smoke_and_keeps_legacy_auc(tmp_path):
    from utils.results import discover

    labels = np.arange(10)
    metrics = calculate(labels, np.eye(10))
    metrics["macro_auc_ovr"] = None
    legacy = tmp_path / "logistic_kair"
    legacy.mkdir()
    (legacy / "metrics.json").write_text(
        json.dumps({"test": metrics}), encoding="utf-8"
    )
    (legacy / "auc.json").write_text(
        json.dumps({"macro_auc_ovr": 1.0, "per_class_auc": [1.0] * 10}),
        encoding="utf-8",
    )
    smoke = tmp_path / "arbitrary"
    smoke.mkdir()
    (smoke / "metrics.json").write_text(
        json.dumps({"config": {"max_batches": 1}, "test": metrics}), encoding="utf-8"
    )
    records = list(discover(tmp_path))
    assert [r["name"] for r in records] == ["logistic_kair"]
    assert records[0]["metrics"]["macro_auc_ovr"] == 1.0
    assert records[0]["metrics"]["per_class"][0]["auc"] == 1.0


def test_smoke_in_tag_does_not_hide_a_full_run(tmp_path):
    from utils.results import discover

    labels = np.arange(10)
    for name, config in [("no_smoke", {"max_batches": 0}), ("legacy_smoke", {})]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "metrics.json").write_text(
            json.dumps({"config": config, "test": calculate(labels, np.eye(10))}),
            encoding="utf-8",
        )
    assert [r["name"] for r in discover(tmp_path)] == ["no_smoke"]


def test_legacy_auc_must_include_every_class(tmp_path):
    from utils.results import discover

    labels = np.arange(10)
    metrics = calculate(labels, np.eye(10))
    metrics["macro_auc_ovr"] = None
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "metrics.json").write_text(
        json.dumps({"test": metrics}), encoding="utf-8"
    )
    (folder / "auc.json").write_text(
        json.dumps({"macro_auc_ovr": 1.0, "per_class_auc": [1.0] * 9}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="per-class AUC count mismatch"):
        list(discover(tmp_path))
