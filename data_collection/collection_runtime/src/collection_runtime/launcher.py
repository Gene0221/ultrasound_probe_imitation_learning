from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from collection_runtime.adapters import (
    Force6DAdapter,
    ImuAdapter,
    PaxiniForceAdapter,
    RealPoseAdapter,
    UltrasoundAdapter,
    VisualPoseAdapter,
)
from collection_runtime.adapters.base import BaseCollectorAdapter
from collection_runtime.config import LauncherConfig, ModuleConfig, load_launcher_config
from collection_runtime.metadata import ModuleStatusRecord, RunMetadataRecorder, SessionMetadataRecorder
from collection_runtime.session_manager import SessionContext, SessionManager
from collection_runtime.state_machine import LauncherState, StateMachine


ADAPTER_REGISTRY = {
    "force6d": Force6DAdapter,
    "imu": ImuAdapter,
    "real_pose": RealPoseAdapter,
    "visual_pose": VisualPoseAdapter,
    "paxini_force": PaxiniForceAdapter,
    "ultrasound": UltrasoundAdapter,
}


@dataclass
class ActiveSession:
    context: SessionContext
    recorder: SessionMetadataRecorder


def read_single_key() -> str:
    if sys.platform.startswith("win"):
        import msvcrt

        while True:
            key = msvcrt.getwch()
            if key in {"\x00", "\xe0"}:
                msvcrt.getwch()
                continue
            if key == "\r":
                return "ENTER"
            if key == "\x03":
                raise KeyboardInterrupt
            return key.lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1)
            if key == "\x1b":
                next_char = sys.stdin.read(1)
                if next_char == "[":
                    sys.stdin.read(1)
                    continue
                continue
            if key in {"\r", "\n"}:
                return "ENTER"
            if key == "\x03":
                raise KeyboardInterrupt
            return key.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class CollectionLauncher:
    def __init__(self, config: LauncherConfig) -> None:
        self.config = config
        self.state_machine = StateMachine()
        self.session_manager = SessionManager(config.output_root)
        self.run_metadata = RunMetadataRecorder(config.logging_root)
        self.adapters = self._build_adapters(config.modules)
        self.active_session: ActiveSession | None = None

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "CollectionLauncher":
        return cls(load_launcher_config(config_path))

    def _build_adapters(self, modules: list[ModuleConfig]) -> list[BaseCollectorAdapter]:
        adapters: list[BaseCollectorAdapter] = []
        for module in modules:
            if not module.enabled:
                continue
            adapter_cls = ADAPTER_REGISTRY.get(module.adapter)
            if adapter_cls is None:
                raise ValueError(f"Unsupported adapter '{module.adapter}' for module '{module.name}'.")
            adapter = adapter_cls(module)
            if hasattr(adapter, "set_control_file"):
                adapter.set_control_file(self.run_metadata.control_dir / f"{module.name}_control.json")
            adapters.append(adapter)
        return adapters

    def initialize(self) -> None:
        module_records: list[ModuleStatusRecord] = []
        for adapter in self.adapters:
            status = adapter.initialize()
            module_record = ModuleStatusRecord(
                name=adapter.name,
                enabled=True,
                initialized=status.initialized,
                placeholder=status.placeholder,
                healthy=status.healthy,
                message=status.message,
            )
            module_records.append(module_record)
            status_label = "OK" if status.healthy else "UNHEALTHY"
            placeholder_suffix = " placeholder" if status.placeholder else ""
            print(f"[{status_label}] {adapter.name}{placeholder_suffix}: {status.message or 'no status message'}")
            if not status.healthy and not self.config.behavior.allow_degraded_start:
                raise RuntimeError(f"Module '{adapter.name}' failed initialization: {status.message}")
        self.run_metadata.set_modules(module_records)

    def run_interactive(self) -> None:
        self.initialize()
        print("Initialization complete.")
        print("Press Enter to start/pause/resume. Press q to quit.")

        try:
            while self.state_machine.state != LauncherState.STOPPED:
                user_input = read_single_key()
                if user_input == self.config.behavior.quit_key.lower():
                    print()
                    self.stop()
                    break
                if user_input == "ENTER":
                    print()
                    if self.state_machine.state in {LauncherState.IDLE, LauncherState.PAUSED}:
                        self.start_or_resume()
                    elif self.state_machine.state == LauncherState.RECORDING:
                        self.pause()
                    continue
        except KeyboardInterrupt:
            print()
            self.stop()

    def start_or_resume(self) -> None:
        was_paused = self.state_machine.state == LauncherState.PAUSED
        self.state_machine.start_or_resume()
        session_context = self.session_manager.create_session()
        session_recorder = SessionMetadataRecorder(session_context.metadata_dir)
        session_recorder.start(
            session_id=session_context.session_id,
            modules=[adapter.describe() for adapter in self.adapters],
        )
        self.run_metadata.add_session(session_context.session_id)
        self.active_session = ActiveSession(context=session_context, recorder=session_recorder)

        for adapter in self.adapters:
            module_dir = session_context.module_dir(adapter.output_subdir)
            if was_paused:
                adapter.resume_recording(module_dir)
            else:
                adapter.start_recording(module_dir)

        print(f"Recording {session_context.session_id}")

    def pause(self) -> None:
        if self.active_session is None:
            raise RuntimeError("No active session to pause.")
        for adapter in self.adapters:
            adapter.pause_recording()
        self.active_session.recorder.finalize(status="paused")
        print(f"Paused {self.active_session.context.session_id}")
        self.active_session = None
        self.state_machine.pause()

    def stop(self) -> None:
        last_session_id = self.run_metadata.session_ids[-1] if self.run_metadata.session_ids else None
        if self.active_session is not None:
            for adapter in self.adapters:
                adapter.pause_recording()
            self.active_session.recorder.finalize(status="stopped")
            print(f"Closed {self.active_session.context.session_id}")
            self.active_session = None

        for adapter in self.adapters:
            adapter.stop()

        self.state_machine.stop()
        self.run_metadata.finish()
        if last_session_id is not None:
            last_session_root = self.session_manager.output_root / last_session_id
            print("Final output directories:")
            for adapter in self.adapters:
                module_dir = last_session_root / adapter.output_subdir
                print(f"  {adapter.name}: {module_dir}")
        print("Collection launcher stopped.")
