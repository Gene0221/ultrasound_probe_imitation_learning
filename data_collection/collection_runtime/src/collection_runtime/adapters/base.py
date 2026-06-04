from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import subprocess

from collection_runtime.config import ModuleConfig
from collection_runtime.metadata import write_json


@dataclass
class ModuleRuntimeStatus:
    initialized: bool
    healthy: bool
    placeholder: bool
    message: str | None = None


class BaseCollectorAdapter(ABC):
    def __init__(self, config: ModuleConfig) -> None:
        self.config = config
        self.status = ModuleRuntimeStatus(
            initialized=False,
            healthy=False,
            placeholder=False,
            message=None,
        )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def output_subdir(self) -> str:
        return self.config.output_subdir

    @abstractmethod
    def initialize(self) -> ModuleRuntimeStatus:
        raise NotImplementedError

    @abstractmethod
    def start_recording(self, session_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def pause_recording(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def resume_recording(self, session_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def health_check(self) -> ModuleRuntimeStatus:
        return self.status

    def run_initialize_command(self) -> None:
        if self.config.initialize_command is None:
            return
        if self.config.workdir is None:
            raise RuntimeError(f"Module '{self.name}' has an initialize_command but no workdir.")
        subprocess.run(
            self.config.initialize_command,
            cwd=self.config.workdir,
            check=True,
        )

    def write_module_event(self, session_dir: Path, event: str, extra: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = {
            "module": self.name,
            "event": event,
            "workdir": str(self.config.workdir) if self.config.workdir else None,
            "initialize_command": self.config.initialize_command,
            "command": self.config.command,
        }
        if extra:
            payload.update(extra)
        write_json(session_dir / f"{event}.json", payload)

    def describe(self) -> dict[str, str | bool | None]:
        return {
            "name": self.name,
            "output_subdir": self.output_subdir,
            "initialized": self.status.initialized,
            "healthy": self.status.healthy,
            "placeholder": self.status.placeholder,
            "message": self.status.message,
        }
