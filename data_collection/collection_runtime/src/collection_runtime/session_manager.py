from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SESSION_PATTERN = re.compile(r"session_(\d{4,})$")


@dataclass
class SessionContext:
    session_id: str
    session_index: int
    root_dir: Path
    metadata_dir: Path

    def module_dir(self, module_subdir: str) -> Path:
        path = self.root_dir / module_subdir
        path.mkdir(parents=True, exist_ok=True)
        return path


class SessionManager:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._next_index = self._scan_next_index()

    def _scan_next_index(self) -> int:
        max_index = 0
        for path in self.output_root.iterdir():
            if not path.is_dir():
                continue
            match = SESSION_PATTERN.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    def create_session(self) -> SessionContext:
        session_index = self._next_index
        session_id = f"session_{session_index:04d}"
        session_root = self.output_root / session_id
        metadata_dir = session_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        self._next_index += 1
        return SessionContext(
            session_id=session_id,
            session_index=session_index,
            root_dir=session_root,
            metadata_dir=metadata_dir,
        )
