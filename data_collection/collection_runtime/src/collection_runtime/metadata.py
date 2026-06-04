from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


@dataclass
class ModuleStatusRecord:
    name: str
    enabled: bool
    initialized: bool = False
    placeholder: bool = False
    healthy: bool = False
    message: str | None = None


@dataclass
class SessionRecord:
    session_id: str
    start_time_utc: str
    end_time_utc: str | None = None
    status: str = "recording"
    modules: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None


class RunMetadataRecorder:
    def __init__(self, logging_root: Path) -> None:
        self.logging_root = ensure_dir(logging_root)
        self.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        self.run_dir = ensure_dir(self.logging_root / self.run_id)
        self.control_dir = ensure_dir(self.run_dir / "controls")
        self.run_summary_path = self.run_dir / "run_summary.json"
        self.run_started_at = utc_now_iso()
        self.run_finished_at: str | None = None
        self.session_ids: list[str] = []
        self.module_statuses: list[ModuleStatusRecord] = []
        self.notes: list[str] = []

    def set_modules(self, module_statuses: list[ModuleStatusRecord]) -> None:
        self.module_statuses = module_statuses
        self.flush()

    def add_session(self, session_id: str) -> None:
        self.session_ids.append(session_id)
        self.flush()

    def add_note(self, note: str) -> None:
        self.notes.append(note)
        self.flush()

    def finish(self) -> None:
        self.run_finished_at = utc_now_iso()
        self.flush()

    def flush(self) -> None:
        write_json(
            self.run_summary_path,
            {
                "run_id": self.run_id,
                "run_started_at_utc": self.run_started_at,
                "run_finished_at_utc": self.run_finished_at,
                "sessions": self.session_ids,
                "modules": [asdict(item) for item in self.module_statuses],
                "notes": self.notes,
            },
        )


class SessionMetadataRecorder:
    def __init__(self, session_metadata_dir: Path) -> None:
        self.session_metadata_path = session_metadata_dir / "session.json"

    def start(self, session_id: str, modules: list[dict[str, Any]]) -> None:
        payload = SessionRecord(
            session_id=session_id,
            start_time_utc=utc_now_iso(),
            modules=modules,
        )
        write_json(self.session_metadata_path, asdict(payload))

    def finalize(self, status: str, error_message: str | None = None) -> None:
        with self.session_metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["end_time_utc"] = utc_now_iso()
        payload["status"] = status
        payload["error_message"] = error_message
        write_json(self.session_metadata_path, payload)
