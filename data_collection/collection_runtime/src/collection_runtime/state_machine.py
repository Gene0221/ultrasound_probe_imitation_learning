from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LauncherState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class StateMachine:
    state: LauncherState = LauncherState.IDLE

    def start_or_resume(self) -> None:
        if self.state not in {LauncherState.IDLE, LauncherState.PAUSED}:
            raise RuntimeError(f"Cannot start or resume from state '{self.state.value}'.")
        self.state = LauncherState.RECORDING

    def pause(self) -> None:
        if self.state != LauncherState.RECORDING:
            raise RuntimeError(f"Cannot pause from state '{self.state.value}'.")
        self.state = LauncherState.PAUSED

    def stop(self) -> None:
        self.state = LauncherState.STOPPED
