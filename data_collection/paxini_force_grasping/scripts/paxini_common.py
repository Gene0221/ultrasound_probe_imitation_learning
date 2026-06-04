from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Union
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
MODULE_PATH = SCRIPT_DIR / "DP-S2015-Elite.py"


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    config_path = Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected config object in {config_path}")
    return payload


def resolve_workspace_path(path_value: Union[str, Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def list_available_ports_details() -> list[dict[str, Any]]:
    from serial.tools import list_ports

    rows: list[dict[str, Any]] = []
    for port in list_ports.comports():
        rows.append(
            {
                "device": port.device,
                "description": getattr(port, "description", None),
                "serial_number": getattr(port, "serial_number", None),
                "manufacturer": getattr(port, "manufacturer", None),
                "product": getattr(port, "product", None),
                "hwid": getattr(port, "hwid", None),
                "location": getattr(port, "location", None),
                "vid": getattr(port, "vid", None),
                "pid": getattr(port, "pid", None),
            }
        )
    return rows


def format_port_info(port_info: dict[str, Any]) -> str:
    return (
        f"{port_info.get('device')}: desc={port_info.get('description')!r}, "
        f"hwid={port_info.get('hwid')!r}, vid={port_info.get('vid')}, pid={port_info.get('pid')}, "
        f"serial={port_info.get('serial_number')!r}, location={port_info.get('location')!r}, "
        f"manufacturer={port_info.get('manufacturer')!r}, product={port_info.get('product')!r}"
    )


def print_ports() -> None:
    rows = list_available_ports_details()
    if not rows:
        print("no serial ports found")
        return
    print("serial ports:")
    for row in rows:
        print(f"  {format_port_info(row)}")


def normalize_serial(value: Any) -> str:
    return "" if value is None else str(value)


def resolve_port(config: dict[str, Any], default_port: str) -> str:
    serial_cfg = config.get("serial", {})
    requested = str(serial_cfg.get("port", default_port)).strip()
    requested_serial = str(serial_cfg.get("serial_number", "") or "").strip()
    port_rows = list_available_ports_details()
    if requested_serial:
        target_serial = normalize_serial(requested_serial).lower()
        matching = [
            row
            for row in port_rows
            if normalize_serial(row.get("serial_number")).lower() == target_serial
        ]
        if len(matching) == 1:
            selected = str(matching[0]["device"])
            print(f"matched serial_number {requested_serial} -> {selected}")
            return selected
        if not matching:
            raise RuntimeError(
                f"Configured serial_number '{requested_serial}' was not found. Available ports: {port_rows}"
            )
        raise RuntimeError(
            f"Configured serial_number '{requested_serial}' matched multiple ports. Available ports: {port_rows}"
        )
    return requested


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
    dp.PORT = resolve_port(config, dp.PORT)
    dp.BAUDRATE = int(serial_cfg.get("baudrate", dp.BAUDRATE))
    dp.SAMPLE_RATE_HZ = float(stream_cfg.get("sampling_hz", dp.SAMPLE_RATE_HZ))
    dp.ZERO_CHECK_SAMPLES = int(calibration_cfg.get("tare_samples", dp.ZERO_CHECK_SAMPLES))
    calibration_path = resolve_workspace_path(calibration_cfg.get("file", "config/dp_s2015_calibration.json"))
    dp.CALIBRATION_FILE = calibration_path
    return calibration_path
