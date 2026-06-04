from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml

from collection_runtime.adapters.base import BaseCollectorAdapter, ModuleRuntimeStatus
from collection_runtime.metadata import ensure_dir


class ControlledProcessAdapter(BaseCollectorAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.control_file: Path | None = None
        self.process: subprocess.Popen[str] | None = None
        self.resolved_initialize_command: list[str] | None = None
        self.resolved_command: list[str] | None = None

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

    def _deep_merge(self, base: Any, override: Any) -> Any:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = dict(base)
            for key, value in override.items():
                merged[key] = self._deep_merge(merged.get(key), value)
            return merged
        return override

    def _resolve_config_path(self, config_arg: str) -> Path:
        config_path = Path(config_arg)
        if config_path.is_absolute():
            return config_path.resolve()
        if self.config.workdir is None:
            raise RuntimeError(f"Module '{self.name}' has a relative config path but no workdir.")
        return (self.config.workdir / config_path).resolve()

    def _build_command_with_overrides(self, command: list[str] | None) -> list[str] | None:
        if command is None:
            return None
        if not self.config.config_overrides:
            return list(command)

        try:
            config_index = command.index("--config")
        except ValueError:
            return list(command)
        if config_index + 1 >= len(command):
            raise RuntimeError(f"Module '{self.name}' command has '--config' without a config path.")
        if self.control_file is None:
            raise RuntimeError(f"Module '{self.name}' requires a control file before resolving overrides.")

        base_config_path = self._resolve_config_path(command[config_index + 1])
        with base_config_path.open("r", encoding="utf-8") as handle:
            base_payload = yaml.safe_load(handle) or {}
        if not isinstance(base_payload, dict):
            raise ValueError(f"Module '{self.name}' base config must be a YAML mapping: {base_config_path}")

        merged_payload = self._deep_merge(base_payload, self.config.config_overrides)
        runtime_config_path = self.control_file.parent / f"{self.name}_runtime_config.yaml"
        runtime_config_path.write_text(
            yaml.safe_dump(merged_payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

        resolved_command = list(command)
        resolved_command[config_index + 1] = str(runtime_config_path)
        return resolved_command

    def _spawn_process(self) -> None:
        if self.resolved_command is None:
            return
        if self.config.workdir is None:
            raise RuntimeError(f"Module '{self.name}' command requires a workdir.")
        if self.control_file is None:
            raise RuntimeError(f"Module '{self.name}' command requires a control file.")

        command = [*self.resolved_command, "--control-file", str(self.control_file)]
        self.process = subprocess.Popen(
            command,
            cwd=self.config.workdir,
            text=True,
        )

    def initialize(self) -> ModuleRuntimeStatus:
        workdir_exists = self.config.workdir is not None and self.config.workdir.exists()
        self.resolved_initialize_command = self._build_command_with_overrides(self.config.initialize_command)
        self.resolved_command = self._build_command_with_overrides(self.config.command)
        if self.resolved_initialize_command is not None:
            self.run_initialize_command(self.resolved_initialize_command)
        if self.resolved_command is not None:
            self._spawn_process()
        self.status = ModuleRuntimeStatus(
            initialized=True,
            healthy=workdir_exists,
            placeholder=self.resolved_command is None,
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
