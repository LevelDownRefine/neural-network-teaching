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
    widget.controls["threshold"].spin.setValue(-45)
    widget.timer.stop()
    restarted = []
    monkeypatch.setattr(widget, "start_simulation", lambda: restarted.append(True))
    widget.finish_simulation(previous_revision, {"wrong": True}, "")
    assert widget.result is previous_result
    assert not widget.export.isEnabled()
    # 修改参数但尚未取得新结果时，重绘仍使用旧结果对应的阈值。
    widget.redraw()
    assert widget.axes[1].lines[1].get_ydata()[0] == -50
    with pytest.raises(ValueError, match="等待"):
        widget.save_results(None)
    assert restarted == [True]


def test_spin_box_preserves_values_between_slider_steps(window):
    widget, _ = window
    duration = widget.controls["duration"]
    duration.spin.setValue(2000)
    duration.spin.setValue(250)
    assert duration.spin.value() == 250
    assert duration.slider.value() == 2
    duration.spin.setValue(200)
    duration.spin.setValue(250)
    assert duration.spin.value() == 250
    duration.slider.setValue(3)
    assert duration.spin.value() == 400
    widget.timer.stop()


@pytest.mark.parametrize("suffix", [".json", ".png"])
@pytest.mark.parametrize("accept", [False, True])
def test_export_confirms_sidecar_collisions(
    window, tmp_path, monkeypatch, suffix, accept
):
    from lif.desktop import QFileDialog, QMessageBox

    widget, app = window
    calculate(widget, app)
    target = tmp_path / "result.csv"
    sidecar = target.with_suffix(suffix)
    sidecar.write_bytes(b"original")
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "CSV")
    )
    confirmations = []

    def confirm(parent, title, message, *args):
        # 所有覆盖确认完成前不能写 CSV。
        assert not target.exists()
        assert sidecar.read_bytes() == b"original"
        confirmations.append(message)
        return (
            QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.No
        )

    monkeypatch.setattr(QMessageBox, "question", confirm)
    widget.export_results()
    assert len(confirmations) == 1
    assert str(sidecar) in confirmations[0]
    assert target.exists() == accept
    if not accept:
        assert sidecar.read_bytes() == b"original"
