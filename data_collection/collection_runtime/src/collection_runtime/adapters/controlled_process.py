from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from collection_runtime.adapters.base import BaseCollectorAdapter, ModuleRuntimeStatus
from collection_runtime.metadata import ensure_dir


class ControlledProcessAdapter(BaseCollectorAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.control_file: Path | None = None
        self.process: subprocess.Popen[str] | None = None

    def set_control_file(self, control_file: Path) -> None:
        self.control_file = control_file
        ensure_dir(control_file.parent)
        self._write_control_state(recording=False, output_dir=None, shutdown=False)

    def _write_control_state(self, recording: bool, output_dir: Path | None, shutdown: bool) -> None:
        if self.control_file is None:
            raise RuntimeError(f"Control file is not configured for module '{self.name}'.")
        payload = {
            "recording": recording,
            "output_dir": str(output_dir) if output_dir is not None else None,
            "shutdown": shutdown,
            "updated_at_monotonic_s": time.monotonic(),
        }
        temp_path = self.control_file.with_name(f"{self.control_file.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp_path, self.control_file)

    def _spawn_process(self) -> None:
        if self.config.command is None:
            return
        if self.config.workdir is None:
            raise RuntimeError(f"Module '{self.name}' command requires a workdir.")
        if self.control_file is None:
            raise RuntimeError(f"Module '{self.name}' command requires a control file.")

        command = [*self.config.command, "--control-file", str(self.control_file)]
        self.process = subprocess.Popen(
            command,
            cwd=self.config.workdir,
            text=True,
        )

    def initialize(self) -> ModuleRuntimeStatus:
        workdir_exists = self.config.workdir is not None and self.config.workdir.exists()
        if self.config.initialize_command is not None:
            self.run_initialize_command()
        if self.config.command is not None:
            self._spawn_process()
        self.status = ModuleRuntimeStatus(
            initialized=True,
            healthy=workdir_exists,
            placeholder=self.config.command is None,
            message="Workspace detected." if workdir_exists else "Workspace path is missing.",
        )
        return self.status

    def start_recording(self, session_dir: Path) -> None:
        self._write_control_state(recording=True, output_dir=session_dir, shutdown=False)
        self.write_module_event(session_dir, "start", {"control_file": str(self.control_file) if self.control_file else None})

    def pause_recording(self) -> None:
        self._write_control_state(recording=False, output_dir=None, shutdown=False)

    def resume_recording(self, session_dir: Path) -> None:
        self._write_control_state(recording=True, output_dir=session_dir, shutdown=False)
        self.write_module_event(session_dir, "resume", {"control_file": str(self.control_file) if self.control_file else None})

    def stop(self) -> None:
        if self.control_file is not None:
            self._write_control_state(recording=False, output_dir=None, shutdown=True)
        if self.process is not None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    break
                time.sleep(0.1)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.process = None

    def health_check(self) -> ModuleRuntimeStatus:
        if self.process is not None and self.process.poll() is not None:
            self.status.healthy = False
            self.status.message = f"Collector process exited with code {self.process.returncode}."
        return self.status
