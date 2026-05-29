from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"
MODULE_PATH = SCRIPT_DIR / "DP-S2015-Elite.py"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected config object in {config_path}")
    return payload


def resolve_workspace_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_dp_module():
    spec = importlib.util.spec_from_file_location("dp_s2015_elite", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def configure_dp_module(dp: Any, config: dict[str, Any]) -> Path:
    serial_cfg = config.get("serial", {})
    stream_cfg = config.get("stream", {})
    calibration_cfg = config.get("calibration", {})
    dp.PORT = str(serial_cfg.get("port", dp.PORT))
    dp.BAUDRATE = int(serial_cfg.get("baudrate", dp.BAUDRATE))
    dp.SAMPLE_RATE_HZ = float(stream_cfg.get("sampling_hz", dp.SAMPLE_RATE_HZ))
    dp.ZERO_CHECK_SAMPLES = int(calibration_cfg.get("tare_samples", dp.ZERO_CHECK_SAMPLES))
    calibration_path = resolve_workspace_path(calibration_cfg.get("file", "config/dp_s2015_calibration.json"))
    dp.CALIBRATION_FILE = calibration_path
    return calibration_path
