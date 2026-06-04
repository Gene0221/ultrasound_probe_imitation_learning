from __future__ import annotations

from pathlib import Path

from hospital_data_collection.adapters.base import BaseCollectorAdapter, ModuleRuntimeStatus
class PlaceholderAdapter(BaseCollectorAdapter):
    def initialize(self) -> ModuleRuntimeStatus:
        self.status = ModuleRuntimeStatus(
            initialized=True,
            healthy=True,
            placeholder=True,
            message=self.config.placeholder_reason or "Placeholder adapter is active.",
        )
        return self.status

    def _write_marker(self, session_dir: Path, event: str) -> None:
        self.write_module_event(
            session_dir,
            event,
            {
                "placeholder": True,
                "message": self.status.message,
            },
        )

    def start_recording(self, session_dir: Path) -> None:
        self._write_marker(session_dir, "start")

    def pause_recording(self) -> None:
        return

    def resume_recording(self, session_dir: Path) -> None:
        self._write_marker(session_dir, "resume")

    def stop(self) -> None:
        return
