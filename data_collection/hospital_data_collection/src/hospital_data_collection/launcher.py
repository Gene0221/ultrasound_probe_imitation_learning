from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hospital_data_collection.adapters import (
    Force6DAdapter,
    ImuAdapter,
    PaxiniForceAdapter,
    RealPoseAdapter,
    UltrasoundAdapter,
    VisualPoseAdapter,
)
from hospital_data_collection.adapters.base import BaseCollectorAdapter
from hospital_data_collection.config import LauncherConfig, ModuleConfig, load_launcher_config
from hospital_data_collection.metadata import ModuleStatusRecord, RunMetadataRecorder, SessionMetadataRecorder
from hospital_data_collection.session_manager import SessionContext, SessionManager
from hospital_data_collection.state_machine import LauncherState, StateMachine


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
            if not status.healthy and not self.config.behavior.allow_degraded_start:
                raise RuntimeError(f"Module '{adapter.name}' failed initialization: {status.message}")
        self.run_metadata.set_modules(module_records)

    def run_interactive(self) -> None:
        self.initialize()
        print("Initialization complete.")
        print("Press Enter to start/pause/resume. Press q then Enter to quit.")

        while self.state_machine.state != LauncherState.STOPPED:
            user_input = input("> ").strip().lower()
            if user_input == self.config.behavior.quit_key.lower():
                self.stop()
                break
            if user_input == "":
                if self.state_machine.state in {LauncherState.IDLE, LauncherState.PAUSED}:
                    self.start_or_resume()
                elif self.state_machine.state == LauncherState.RECORDING:
                    self.pause()
                continue
            print("Unsupported input. Use Enter to toggle recording or q to quit.")

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
        print("Collection launcher stopped.")
