import json
from argparse import ArgumentTypeError
from pathlib import Path

import pytest
from utils.cli import positive_int
from utils.io import atomic_write_json


def test_atomic_json_keeps_previous_result_on_interruption(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(self, target):
        raise OSError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError):
            atomic_write_json(path, {"new": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    atomic_write_json(path, {"new": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_epochs_must_be_positive(value):
    with pytest.raises(ArgumentTypeError):
        positive_int(value)
