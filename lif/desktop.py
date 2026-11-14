"""运行：uv run --extra cpu --extra gui python -m lif.desktop。"""

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSignalBlocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lif.model import SimulationParameters, simulate_experiment
from lif.plot_lif import plot_signal


class WorkerSignals(QObject):
    finished = Signal(int, object, str)


class SimulationWorker(QRunnable):
    def __init__(self, revision, parameters):
        super().__init__()
        self.revision = revision
        self.parameters = parameters
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            data = simulate_experiment(self.parameters)
            self.signals.finished.emit(self.revision, data, "")
        except Exception as error:
            self.signals.finished.emit(self.revision, None, str(error))


class ParameterControl(QWidget):
    changed = Signal()

    def __init__(self, low, high, step, value):
        super().__init__()
        self.step = step
        self.low = low
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round((high - low) / step))
        self.slider.setTracking(False)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(low, high)
        self.spin.setDecimals(len(str(step).split(".")[1]) if "." in str(step) else 0)
        self.spin.setSingleStep(step)
        self.spin.setValue(value)
        self.slider.setValue(round((value - low) / step))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(
            lambda index: self.spin.setValue(low + index * step)
        )
        self.spin.valueChanged.connect(self._sync_slider)
        self.spin.valueChanged.connect(self.changed)

    def _sync_slider(self, number):
        # 只移动滑块指示位置，不用近似步长反向改写用户输入。
        with QSignalBlocker(self.slider):
            self.slider.setValue(round((number - self.low) / self.step))


class LIFWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LIF 神经元实验")
        self.resize(1300, 900)
        self.parameters = SimulationParameters()
        self.result = None
        self.result_parameters = None
        self.revision = 0
        self.busy = False
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.start_simulation)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        panel = QWidget()
        form = QFormLayout(panel)
        self.mode = QComboBox()
        self.mode.addItem("恒定电流", "constant")
        self.mode.addItem("正弦电流", "sinusoidal")
        form.addRow("输入方式", self.mode)
        self.controls = {}
        settings = [
            ("current", "基础电流 (nA)", 0, 4, 0.1),
            ("amplitude", "振幅 (nA)", 0, 3, 0.1),
            ("frequency", "频率 (Hz)", 0, 50, 0.5),
            ("phase", "相位 (rad)", -3.14, 3.14, 0.01),
            ("threshold", "发放阈值 (mV)", -60, -40, 1),
            ("tau", "膜时间常数 (ms)", 5, 50, 1),
            ("resistance", "膜电阻 (MΩ)", 1, 30, 1),
            ("refractory", "不应期 (ms)", 0, 10, 0.1),
            ("duration", "仿真时长 (ms)", 100, 5000, 100),
        ]
        for key, label, low, high, step in settings:
            control = ParameterControl(low, high, step, getattr(self.parameters, key))
            control.changed.connect(self.request_simulation)
            self.controls[key] = control
            form.addRow(label, control)
        self.view = QComboBox()
        for text, value in [("前 200 ms", 200), ("前 500 ms", 500), ("全部", 0)]:
            self.view.addItem(text, value)
        self.view.currentIndexChanged.connect(self.redraw)
        form.addRow("显示范围", self.view)
        form.addRow(
            QLabel(
                "静息电位 −65 mV · 时间步 0.1 ms\n发放后回到静息电位；松开滑块后计算。"
            )
        )
        buttons = QHBoxLayout()
        reset = QPushButton("重置参数")
        reset.clicked.connect(self.reset_parameters)
        buttons.addWidget(reset)
        self.export = QPushButton("导出结果")
        self.export.setEnabled(False)
        self.export.clicked.connect(self.export_results)
        buttons.addWidget(self.export)
        form.addRow(buttons)
        self.statistics = QLabel("等待仿真")
        self.statistics.setWordWrap(True)
        form.addRow(self.statistics)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(480)
        layout.addWidget(scroll)
        graph = QGroupBox("输入与响应")
        graph_layout = QVBoxLayout(graph)
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.subplots(4, 1, sharex=True)
        graph_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        graph_layout.addWidget(self.canvas)
        layout.addWidget(graph, 1)
        self.mode.currentIndexChanged.connect(self.request_simulation)
        self.request_simulation()

    def read_parameters(self):
        values = {key: control.spin.value() for key, control in self.controls.items()}
        return SimulationParameters(mode=self.mode.currentData(), **values)

    @Slot()
    def request_simulation(self):
        self.revision += 1
        self.export.setEnabled(False)
        for key in ("amplitude", "frequency", "phase"):
            self.controls[key].setEnabled(self.mode.currentData() == "sinusoidal")
        self.statusBar().showMessage(
            "等待计算…" if not self.busy else "正在计算，完成后更新最新参数…"
        )
        self.timer.start()

    @Slot()
    def start_simulation(self):
        if self.busy:
            return
        self.timer.stop()
        self.busy = True
        self.parameters = self.read_parameters()
        self.worker = SimulationWorker(self.revision, self.parameters)
        self.worker.signals.finished.connect(self.finish_simulation)
        self.statusBar().showMessage("正在仿真…")
        self.pool.start(self.worker)

    @Slot(int, object, str)
    def finish_simulation(self, revision, result, error):
        self.busy = False
        if revision != self.revision:
            self.start_simulation()
            return
        if error:
            self.statusBar().showMessage(f"计算失败：{error}")
            return
        self.result = result
        self.result_parameters = self.parameters
        self.redraw()
        count = int(result["spikes"].sum())
        indices = np.flatnonzero(result["spikes"])
        first = f"{result['time_ms'][indices[0]]:.1f} ms" if len(indices) else "未发放"
        self.statistics.setText(
            f"脉冲总数：{count}\n平均发放率：{count / (self.parameters.duration / 1000):.2f} Hz\n"
            f"首个脉冲：{first}\n统计范围：{self.parameters.duration:g} ms"
        )
        self.export.setEnabled(True)
        self.statusBar().showMessage("计算完成")

    @Slot()
    def redraw(self):
        if self.result is None:
            return
        for ax, signal in zip(
            self.axes, ("current", "voltage", "spikes", "rate"), strict=True
        ):
            ax.clear()
            plot_signal(
                ax,
                signal,
                self.result,
                self.result_parameters.rest,
                self.result_parameters.threshold,
            )
        span = self.view.currentData() or self.result_parameters.duration
        self.axes[-1].set(
            xlabel="Time (ms)", xlim=(0, min(span, self.result_parameters.duration))
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @Slot()
    def reset_parameters(self):
        defaults = SimulationParameters()
        for key, control in self.controls.items():
            control.spin.setValue(getattr(defaults, key))
        self.mode.setCurrentIndex(0)
        self.view.setCurrentIndex(0)
        self.request_simulation()

    @Slot()
    def export_results(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            "lif-result.csv",
            "CSV (*.csv)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not filename:
            return
        path = Path(filename).with_suffix(".csv")
        collisions = [
            target
            for target in (path, path.with_suffix(".json"), path.with_suffix(".png"))
            if target.exists()
        ]
        if collisions:
            answer = QMessageBox.question(
                self,
                "确认覆盖",
                "以下文件已存在，是否覆盖？\n"
                + "\n".join(str(target) for target in collisions),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.save_results(path)
        except OSError as error:
            QMessageBox.warning(self, "导出失败", str(error))

    def save_results(self, path):
        if self.result is None or not self.export.isEnabled():
            raise ValueError("请等待当前参数计算完成")
        data = self.result
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "input_time_ms",
                    "time_ms",
                    "current_nA",
                    "voltage_mV",
                    "spike",
                    "rate_Hz",
                )
            )
            writer.writerows(
                zip(
                    *(
                        data[key]
                        for key in (
                            "input_time_ms",
                            "time_ms",
                            "current",
                            "voltage",
                            "spikes",
                            "rate",
                        )
                    ),
                    strict=True,
                )
            )
        path.with_suffix(".json").write_text(
            json.dumps(asdict(self.result_parameters), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        self.figure.savefig(path.with_suffix(".png"), dpi=150)
        self.statusBar().showMessage(f"已导出 CSV、参数 JSON 和图片：{path.parent}")

    def closeEvent(self, event):
        self.timer.stop()
        self.pool.waitForDone()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = LIFWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
