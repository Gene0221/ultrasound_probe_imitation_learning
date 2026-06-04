from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BehaviorConfig:
    allow_degraded_start: bool
    start_pause_key: str
    quit_key: str


@dataclass
class ModuleConfig:
    name: str
    enabled: bool
    adapter: str
    allow_placeholder: bool
    output_subdir: str
    initialize_command: list[str] | None
    command: list[str] | None
    workdir: Path | None
    placeholder_reason: str | None = None


@dataclass
class LauncherConfig:
    project_root: Path
    config_path: Path
    output_root: Path
    logging_root: Path
    behavior: BehaviorConfig
    modules: list[ModuleConfig]


def _ensure_list_or_none(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("Module command must be null or a list of strings.")


def _resolve_optional_path(project_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def load_launcher_config(config_path: str | Path) -> LauncherConfig:
    resolved_config_path = Path(config_path).resolve()
    project_root = resolved_config_path.parent.parent
    with resolved_config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Launcher config must be a YAML mapping.")

    behavior_payload = payload.get("behavior", {})
    modules_payload = payload.get("modules", {})
    if not isinstance(modules_payload, dict):
        raise ValueError("'modules' must be a mapping.")

    modules: list[ModuleConfig] = []
    for module_name, module_payload in modules_payload.items():
        if not isinstance(module_payload, dict):
            raise ValueError(f"Module '{module_name}' config must be a mapping.")
        modules.append(
            ModuleConfig(
                name=module_name,
                enabled=bool(module_payload.get("enabled", True)),
                adapter=str(module_payload.get("adapter", module_name)),
                allow_placeholder=bool(module_payload.get("allow_placeholder", False)),
                output_subdir=str(module_payload.get("output_subdir", module_name)),
                initialize_command=_ensure_list_or_none(module_payload.get("initialize_command")),
                command=_ensure_list_or_none(module_payload.get("command")),
                workdir=_resolve_optional_path(project_root, module_payload.get("workdir")),
                placeholder_reason=module_payload.get("placeholder_reason"),
            )
        )

    return LauncherConfig(
        project_root=project_root,
        config_path=resolved_config_path,
        output_root=_resolve_optional_path(project_root, payload.get("output_root")) or project_root / "output",
        logging_root=_resolve_optional_path(project_root, payload.get("logging_root")) or project_root / "logs",
        behavior=BehaviorConfig(
            allow_degraded_start=bool(behavior_payload.get("allow_degraded_start", True)),
            start_pause_key=str(behavior_payload.get("start_pause_key", "ENTER")),
            quit_key=str(behavior_payload.get("quit_key", "q")),
        ),
        modules=modules,
    )
