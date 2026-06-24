from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
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


class HospitalCollectionWindow(QMainWindow):
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self._controller = controller

        self.setWindowTitle("Hospital Collection Console")
        self.resize(1400, 920)

        self.latest_status: dict[str, Any] = {}
        self._backend_initialized = False
        self._prev_state: str | None = None

        # UI state tracking
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

        self.launch_button.clicked.connect(self._launch_backend)
        self.start_button.clicked.connect(self._on_start_session)
        self.pause_button.clicked.connect(self._on_pause_session)
        self.resume_button.clicked.connect(self._on_resume_session)
        self.stop_button.clicked.connect(self._shutdown_backend)

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

    def _on_start_session(self) -> None:
        self._append_log("[APP] Sent command: start session")
        self._controller.start_or_resume()

    def _on_pause_session(self) -> None:
        self._append_log("[APP] Sent command: pause session")
        self._controller.pause()

    def _on_resume_session(self) -> None:
        self._append_log("[APP] Sent command: resume session")
        self._controller.start_or_resume()

    def _launch_backend(self) -> None:
        if self._backend_initialized:
            return
        success, message = self._controller.initialize()
        if success:
            self._backend_initialized = True
            self._controller.log_message.connect(self._append_log)
            self._controller.status_updated.connect(self._on_status_updated)
            self._append_log("[BACKEND] Backend initialised successfully.")
        else:
            self._append_log(f"[BACKEND] Initialisation failed: {message}")
            QMessageBox.critical(self, "Backend Error", f"Failed to launch: {message}")
        self._refresh_controls()

    def _shutdown_backend(self) -> None:
        self._controller.shutdown()
        self._backend_initialized = False
        self._append_log("[BACKEND] Backend shut down.")
        self._refresh_controls()

    def _on_status_updated(self, payload: dict[str, Any]) -> None:
        self.latest_status = payload

        new_state = str(payload.get("state", "stopped"))
        if new_state != self._prev_state:
            self._prev_state = new_state
            session_id = payload.get("active_session_id") or payload.get("last_session_id") or ""
            if new_state == "recording":
                self._append_log(f"[BACKEND] Recording {session_id}")
            elif new_state == "paused":
                self._append_log(f"[BACKEND] Paused {session_id}")
            elif new_state == "stopped":
                self._append_log("[BACKEND] Backend stopped.")

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
            widgets.health_label.setText("OK" if healthy else "UNHEALTHY" if healthy is False else "-")
            widgets.message_label.setText(str(module_payload.get("message") or "-"))
            widgets.output_label.setText(str(last_output_dirs.get(module_name, "-")))

        if isinstance(last_output_dirs, dict) and last_output_dirs:
            lines = [f"{name}: {path}" for name, path in last_output_dirs.items()]
            self.paths_text.setPlainText("\n".join(lines))
        else:
            self.paths_text.setPlainText("")
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        state = str(self.latest_status.get("state", "stopped")) if self.latest_status else "stopped"
        self.launch_button.setEnabled(not self._backend_initialized)
        self.start_button.setEnabled(self._backend_initialized and state == "idle")
        self.pause_button.setEnabled(self._backend_initialized and state == "recording")
        self.resume_button.setEnabled(self._backend_initialized and state == "paused")
        self.stop_button.setEnabled(self._backend_initialized)


def run() -> None:
    from hospital_collection.backend_controller import BackendController

    config_path = APP_ROOT / "config" / "default.yaml"
    controller = BackendController(config_path)

    app = QApplication(sys.argv)
    window = HospitalCollectionWindow(controller)
    window.show()
    sys.exit(app.exec())


# Re-export from dataclass for the main window
from dataclasses import dataclass  # noqa: E402


@dataclass
class ModuleWidgets:
    name_label: QLabel
    health_label: QLabel
    message_label: QLabel
    output_label: QLabel
