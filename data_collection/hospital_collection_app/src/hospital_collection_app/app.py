from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


APP_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLECTION_ROOT = APP_ROOT.parent
HOSPITAL_BACKEND_ROOT = DATA_COLLECTION_ROOT / "hospital_data_collection"


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class ModuleWidgets:
    name_label: QLabel
    health_label: QLabel
    message_label: QLabel
    output_label: QLabel


class HospitalCollectionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hospital Collection Console")
        self.resize(1400, 920)

        self.runtime_dir = APP_ROOT / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.command_file = self.runtime_dir / "backend_command.json"
        self.status_file = self.runtime_dir / "backend_status.json"
        self.command_sequence = 0
        self.latest_status: dict[str, Any] = {}

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(
            [
                str(HOSPITAL_BACKEND_ROOT / "main.py"),
                "--command-file",
                str(self.command_file),
                "--status-file",
                str(self.status_file),
            ]
        )
        self.process.setWorkingDirectory(str(HOSPITAL_BACKEND_ROOT))
        self.process.readyReadStandardOutput.connect(self._append_stdout)
        self.process.readyReadStandardError.connect(self._append_stderr)
        self.process.started.connect(self._on_backend_started)
        self.process.finished.connect(self._on_backend_finished)
        self.process.errorOccurred.connect(self._on_backend_error)

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(300)
        self.status_timer.timeout.connect(self._refresh_status)

        self.module_widgets: dict[str, ModuleWidgets] = {}
        self._build_ui()
        self._refresh_controls()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        controls = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls)
        self.launch_button = QPushButton("Launch System")
        self.start_button = QPushButton("Start Session")
        self.pause_button = QPushButton("Pause Session")
        self.resume_button = QPushButton("Resume Session")
        self.stop_button = QPushButton("Quit System")

        self.launch_button.clicked.connect(self._start_backend_process)
        self.start_button.clicked.connect(lambda: self._send_command("start_or_resume"))
        self.pause_button.clicked.connect(lambda: self._send_command("pause"))
        self.resume_button.clicked.connect(lambda: self._send_command("start_or_resume"))
        self.stop_button.clicked.connect(lambda: self._send_command("quit"))

        for button in [
            self.launch_button,
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ]:
            controls_layout.addWidget(button)
        controls_layout.addStretch(1)

        splitter = QSplitter(Qt.Horizontal)

        status_panel = QWidget()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setSpacing(12)

        run_box = QGroupBox("Run Status")
        run_form = QFormLayout(run_box)
        self.state_label = QLabel("stopped")
        self.active_session_label = QLabel("-")
        self.last_session_label = QLabel("-")
        self.output_root_label = QLabel("-")
        self.output_root_label.setWordWrap(True)
        self.run_dir_label = QLabel("-")
        self.run_dir_label.setWordWrap(True)
        run_form.addRow("Backend State", self.state_label)
        run_form.addRow("Active Session", self.active_session_label)
        run_form.addRow("Last Session", self.last_session_label)
        run_form.addRow("Output Root", self.output_root_label)
        run_form.addRow("Run Directory", self.run_dir_label)
        status_layout.addWidget(run_box)

        modules_box = QGroupBox("Module Status")
        modules_layout = QGridLayout(modules_box)
        modules_layout.addWidget(QLabel("Module"), 0, 0)
        modules_layout.addWidget(QLabel("Health"), 0, 1)
        modules_layout.addWidget(QLabel("Message"), 0, 2)
        modules_layout.addWidget(QLabel("Last Output"), 0, 3)
        for row_index, module_name in enumerate(["visual_pose", "paxini_force", "imu", "ultrasound"], start=1):
            name_label = QLabel(module_name)
            health_label = QLabel("-")
            message_label = QLabel("-")
            message_label.setWordWrap(True)
            output_label = QLabel("-")
            output_label.setWordWrap(True)
            modules_layout.addWidget(name_label, row_index, 0)
            modules_layout.addWidget(health_label, row_index, 1)
            modules_layout.addWidget(message_label, row_index, 2)
            modules_layout.addWidget(output_label, row_index, 3)
            self.module_widgets[module_name] = ModuleWidgets(
                name_label=name_label,
                health_label=health_label,
                message_label=message_label,
                output_label=output_label,
            )
        status_layout.addWidget(modules_box)

        paths_box = QGroupBox("Latest Output Directories")
        paths_layout = QVBoxLayout(paths_box)
        self.paths_text = QPlainTextEdit()
        self.paths_text.setReadOnly(True)
        self.paths_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.paths_text.setFont(mono)
        paths_layout.addWidget(self.paths_text)
        status_layout.addWidget(paths_box)
        status_layout.addStretch(1)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_box = QGroupBox("Backend Log")
        log_box_layout = QVBoxLayout(log_box)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_text.setFont(mono)
        log_box_layout.addWidget(self.log_text)
        log_layout.addWidget(log_box)

        splitter.addWidget(status_panel)
        splitter.addWidget(log_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(controls)
        root_layout.addWidget(splitter, 1)

    def _append_log(self, text: str) -> None:
        cleaned = text.rstrip()
        if not cleaned:
            return
        self.log_text.appendPlainText(cleaned)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _append_stdout(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(text)

    def _append_stderr(self) -> None:
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self._append_log(text)

    def _start_backend_process(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            return
        for path in [self.command_file, self.status_file]:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        self.process.start()

    def _on_backend_started(self) -> None:
        self._append_log("[APP] Backend process started.")
        self.status_timer.start()
        self._refresh_controls()

    def _on_backend_finished(self) -> None:
        self.status_timer.stop()
        self._append_log("[APP] Backend process finished.")
        self._refresh_status()
        self._refresh_controls()

    def _on_backend_error(self, _error: QProcess.ProcessError) -> None:
        self._append_log(f"[APP] Backend process error: {self.process.errorString()}")
        self._refresh_controls()

    def _send_command(self, command: str) -> None:
        if self.process.state() == QProcess.NotRunning:
            QMessageBox.warning(self, "Backend Not Running", "Launch the collection system first.")
            return
        self.command_sequence += 1
        payload = {"sequence": self.command_sequence, "command": command}
        self.command_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._append_log(f"[APP] Sent command: {command}")

    def _refresh_status(self) -> None:
        payload = read_json_if_exists(self.status_file)
        if payload is None:
            return
        self.latest_status = payload
        self.state_label.setText(str(payload.get("state", "-")))
        self.active_session_label.setText(str(payload.get("active_session_id") or "-"))
        self.last_session_label.setText(str(payload.get("last_session_id") or "-"))
        self.output_root_label.setText(str(payload.get("output_root") or "-"))
        self.run_dir_label.setText(str(payload.get("run_dir") or "-"))

        module_rows = payload.get("modules", [])
        last_output_dirs = payload.get("last_output_dirs", {})
        module_map = {str(item.get("name")): item for item in module_rows if isinstance(item, dict)}
        for module_name, widgets in self.module_widgets.items():
            module_payload = module_map.get(module_name, {})
            healthy = module_payload.get("healthy")
            widgets.health_label.setText("OK" if healthy else "UNHEALTHY")
            widgets.message_label.setText(str(module_payload.get("message") or "-"))
            widgets.output_label.setText(str(last_output_dirs.get(module_name, "-")))

        if isinstance(last_output_dirs, dict) and last_output_dirs:
            lines = [f"{name}: {path}" for name, path in last_output_dirs.items()]
            self.paths_text.setPlainText("\n".join(lines))
        else:
            self.paths_text.setPlainText("")
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        running = self.process.state() != QProcess.NotRunning
        state = str(self.latest_status.get("state", "stopped")) if self.latest_status else "stopped"
        self.launch_button.setEnabled(not running)
        self.start_button.setEnabled(running and state == "idle")
        self.pause_button.setEnabled(running and state == "recording")
        self.resume_button.setEnabled(running and state == "paused")
        self.stop_button.setEnabled(running)


def run() -> None:
    app = QApplication(sys.argv)
    window = HospitalCollectionWindow()
    window.show()
    sys.exit(app.exec())
