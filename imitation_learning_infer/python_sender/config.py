from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFER_ROOT = PROJECT_ROOT


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def resolve_path(path_value: str | Path, *, base: Path = INFER_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()
