import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from lif.desktop import LIFWindow  # noqa: E402


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    widget = LIFWindow()
    widget.controls["duration"].spin.setValue(100)
    widget.timer.stop()
    yield widget, app
    widget.close()


def calculate(widget, app):
    widget.timer.stop()
    widget.start_simulation()
    deadline = time.monotonic() + 60
    while widget.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not widget.busy, "simulation timed out"
    assert widget.export.isEnabled(), widget.statusBar().currentMessage()


def test_background_simulation_and_export(window, tmp_path):
    import csv
    import json

    widget, app = window
    calculate(widget, app)
    assert len(widget.result["spikes"]) == 1000
    assert np.all(widget.result["current"] == 2)
    widget.mode.setCurrentIndex(1)
    calculate(widget, app)
    assert np.ptp(widget.result["current"]) > 0
    target = tmp_path / "result.csv"
    widget.save_results(target)
    with target.open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1001
    assert json.loads(target.with_suffix(".json").read_text())["mode"] == "sinusoidal"
    assert target.with_suffix(".png").stat().st_size > 0


def test_stale_result_is_not_displayed_or_exported(window, monkeypatch):
    widget, app = window
    calculate(widget, app)
    previous_result = widget.result
    previous_revision = widget.revision
    widget.controls["current"].spin.setValue(3)
    widget.timer.stop()
    restarted = []
    monkeypatch.setattr(widget, "start_simulation", lambda: restarted.append(True))
    widget.finish_simulation(previous_revision, {"wrong": True}, "")
    assert widget.result is previous_result
    assert not widget.export.isEnabled()
    assert restarted == [True]
