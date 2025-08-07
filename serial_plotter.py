"""A simple general-purpose serial data plotter GUI (PyQt5 version).

This tool allows the user to select an available serial port and plots incoming
float values in real time.

Dependencies:
    - pyserial
    - matplotlib
    - PyQt5

Usage:
    python -m tools.serial_plotter
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Matplotlib must choose the Qt backend **before** pyplot is imported.
# ---------------------------------------------------------------------------
import matplotlib  # isort: skip

matplotlib.use("Qt5Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from serial import Serial, SerialException
from serial.tools import list_ports


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class SerialConfig:
    port: str
    baudrate: int = 9600
    timeout: float = 1.0  # seconds


# ---------------------------------------------------------------------------
# Serial Reading Thread
# ---------------------------------------------------------------------------
class SerialReader(threading.Thread):
    """Thread that continuously reads float values from a serial port."""

    def __init__(self, config: SerialConfig, data_queue: queue.Queue[float]):
        super().__init__(daemon=True)
        self.config = config
        self.data_queue = data_queue
        self._stop_event = threading.Event()
        self._serial: Optional[Serial] = None

    def run(self) -> None:
        try:
            self._serial = Serial(
                self.config.port, self.config.baudrate, timeout=self.config.timeout
            )
        except SerialException as exc:
            # We cannot show a Qt dialog from a non-GUI thread, but we can put
            # a sentinel into the queue to signal an error.
            self.data_queue.put(exc)
            return

        # Flush any existing input
        self._serial.flushInput()

        while not self._stop_event.is_set():
            try:
                line: bytes = self._serial.readline()
                if not line:
                    continue  # timed out
                try:
                    value = float(line.strip())
                    self.data_queue.put(value)
                except ValueError:
                    # Ignore malformed lines
                    continue
            except SerialException as exc:
                self.data_queue.put(exc)
                break

        # Clean up
        if self._serial and self._serial.is_open:
            self._serial.close()

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Main Application GUI
# ---------------------------------------------------------------------------
class SerialPlotter(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Serial Data Plotter")
        self.resize(800, 600)

        # Central widget and main layout
        central = QWidget(self)
        self.setCentralWidget(central)
        self._main_layout = QVBoxLayout(central)
        self._main_layout.setContentsMargins(5, 5, 5, 5)

        # Queue for inter-thread communication
        self._data_queue: queue.Queue[float | Exception] = queue.Queue()

        # Data storage
        self._data: List[float] = []
        self._time: List[float] = []
        self._start_time: float = time.time()

        # Serial thread holder
        self._reader: Optional[SerialReader] = None

        # Build UI and plot
        self._create_controls()
        self._setup_plot()

        # Timer to poll queue for new data (approx 30 Hz)
        self._timer = self.startTimer(33)  # 33 ms ~ 30 FPS

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _create_controls(self) -> None:
        control_widget = QWidget(self)
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.addWidget(control_widget, alignment=Qt.AlignTop)

        # Serial port dropdown
        self._port_combo = QComboBox(control_widget)
        self._refresh_ports()
        control_layout.addWidget(self._port_combo)

        # Refresh button
        refresh_btn = QPushButton("Refresh", control_widget)
        refresh_btn.clicked.connect(self._refresh_ports)
        control_layout.addWidget(refresh_btn)

        # Baudrate entry
        control_layout.addWidget(QLabel("Baudrate:", control_widget))
        self._baud_edit = QLineEdit("9600", control_widget)
        self._baud_edit.setFixedWidth(80)
        control_layout.addWidget(self._baud_edit)

        # Spacer
        control_layout.addStretch()

        # Start / Stop buttons
        self._start_btn = QPushButton("Start", control_widget)
        self._start_btn.clicked.connect(self._start_reading)
        control_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop", control_widget)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_reading)
        control_layout.addWidget(self._stop_btn)

    # ---------------------------------------------------------
    # Plot setup
    # ---------------------------------------------------------
    def _setup_plot(self) -> None:
        self._fig, self._ax = plt.subplots(figsize=(8, 4))
        (self._line,) = self._ax.plot([], [], lw=2)
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Value")
        self._ax.set_title("Real-time Serial Data")
        self._ax.grid(True)

        self._canvas = FigureCanvas(self._fig)
        self._main_layout.addWidget(self._canvas, stretch=1)

        # Use FuncAnimation to periodically update the plot
        self._ani = animation.FuncAnimation(
            self._fig, self._update_plot, interval=200, blit=False
        )

    # ---------------------------------------------------------
    # Serial port helpers
    # ---------------------------------------------------------
    @staticmethod
    def _get_serial_ports() -> List[str]:
        return [port.device for port in list_ports.comports()]

    def _refresh_ports(self) -> None:
        ports = self._get_serial_ports()
        self._port_combo.clear()
        self._port_combo.addItems(ports)

    # ---------------------------------------------------------
    # Start / Stop reading
    # ---------------------------------------------------------
    def _start_reading(self) -> None:
        port = self._port_combo.currentText()
        if not port:
            QMessageBox.warning(
                self, "No Port Selected", "Please select a serial port to start."
            )
            return
        try:
            baud = int(self._baud_edit.text())
        except ValueError:
            QMessageBox.critical(
                self, "Invalid Baudrate", "Baudrate must be an integer."
            )
            return

        config = SerialConfig(port=port, baudrate=baud)
        self._reader = SerialReader(config, self._data_queue)
        self._reader.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._start_time = time.time()
        self._data.clear()
        self._time.clear()

    def _stop_reading(self) -> None:
        if self._reader:
            self._reader.stop()
            self._reader.join(timeout=1.0)
            self._reader = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ---------------------------------------------------------
    # Qt timer event: poll data queue
    # ---------------------------------------------------------
    def timerEvent(self, event):  # pylint: disable=invalid-name
        if event.timerId() == self._timer:
            while not self._data_queue.empty():
                item = self._data_queue.get()
                if isinstance(item, Exception):
                    # Display error and stop
                    QMessageBox.critical(self, "Serial Error", str(item))
                    self._stop_reading()
                    continue
                value = item
                current_time = time.time() - self._start_time
                self._time.append(current_time)
                self._data.append(value)

            # Redraw the canvas if new data arrived (animation covers plot update)
            if self._data:
                self._canvas.draw_idle()

    # ---------------------------------------------------------
    # Plot updating (called by FuncAnimation)
    # ---------------------------------------------------------
    def _update_plot(self, *_args):
        if not self._data:
            return (self._line,)

        self._line.set_data(self._time, self._data)
        # Adjust x-axis
        self._ax.set_xlim(max(0, self._time[-1] - 30), self._time[-1] + 1)
        # Adjust y-axis with padding
        ymin = min(self._data)
        ymax = max(self._data)
        padding = 1.0 if ymin == ymax else 0.05 * (ymax - ymin)
        self._ax.set_ylim(ymin - padding, ymax + padding)
        return (self._line,)

    # ---------------------------------------------------------
    # Cleanup on close
    # ---------------------------------------------------------
    def closeEvent(self, event):  # pylint: disable=invalid-name
        self._stop_reading()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = QApplication(sys.argv)
    window = SerialPlotter()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
