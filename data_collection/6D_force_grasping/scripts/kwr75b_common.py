from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path
from typing import Any
import yaml

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyserial is required for 6D force acquisition. Install it with `pip install pyserial`.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FRAME_HEADERS = (b"\x49\xaa", b"\x48\xaa")
FRAME_TAIL = b"\x0d\x0a"
FRAME_LEN = 28


def format_port_info(port_info) -> str:
    return (
        f"{port_info.device}: desc={port_info.description!r}, "
        f"hwid={port_info.hwid!r}, vid={port_info.vid}, pid={port_info.pid}, "
        f"serial={port_info.serial_number!r}, location={port_info.location!r}, "
        f"manufacturer={port_info.manufacturer!r}, product={port_info.product!r}"
    )


def normalize_serial(value: Any) -> str:
    return "" if value is None else str(value)


def print_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("no serial ports found")
        return
    print("serial ports:")
    for port_info in ports:
        print(f"  {format_port_info(port_info)}")


def parse_frame(frame: bytes):
    if len(frame) != FRAME_LEN:
        raise ValueError(f"invalid frame length: {len(frame)}")
    if frame[:2] not in FRAME_HEADERS:
        raise ValueError(f"invalid header: {frame[:2].hex(' ')}")
    if frame[-2:] != FRAME_TAIL:
        raise ValueError(f"invalid tail: {frame[-2:].hex(' ')}")
    return struct.unpack("<ffffff", frame[2:26])


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected config object in {config_path}")
    return payload


def resolve_workspace_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def list_available_ports_details() -> list[dict[str, Any]]:
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


def resolve_port(
    config: dict[str, Any] | None = None,
    port: str | None = None,
    serial_number: str | None = None,
) -> str:
    serial_cfg = (config or {}).get("serial", {})
    requested = str(port if port is not None else serial_cfg.get("port", "AUTO")).strip()
    requested_serial = str(
        serial_number if serial_number is not None else serial_cfg.get("serial_number", "")
    ).strip()
    port_rows = list_available_ports_details()
    ports = [row["device"] for row in port_rows]
    print(f"available ports: {ports}")
    if requested_serial:
        print(f"requested serial_number: {requested_serial}")
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
                f"Configured serial_number '{requested_serial}' was not found. "
                f"Available ports: {port_rows}"
            )
        raise RuntimeError(
            f"Configured serial_number '{requested_serial}' matched multiple ports. "
            f"Available ports: {port_rows}"
        )
    if requested.upper() == "AUTO":
        if len(ports) == 1:
            selected = ports[0]
            print(f"auto selected serial port: {selected}")
            return selected
        if not ports:
            raise RuntimeError("No serial ports detected. Connect the sensor and check the USB/serial adapter.")
        raise RuntimeError(
            "AUTO port selection is ambiguous because multiple serial ports are available: "
            f"{port_rows}. Set `serial.serial_number` or `serial.port` explicitly in config/default.yaml."
        )
    if requested not in ports:
        raise RuntimeError(f"Configured serial port '{requested}' was not found. Available ports: {port_rows}")
    return requested


def read_keyboard_char():
    if not sys.stdin.isatty():
        return None
    if sys.platform.startswith("win"):
        import msvcrt

        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
        return None
    return None


def read_exact_frame(ser: serial.Serial, buffer: bytearray, debug=False):
    last_status_time = time.monotonic()
    while True:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buffer.extend(chunk)
            if debug:
                print(f"rx {len(chunk)} bytes: {chunk.hex(' ')}")
        elif debug and time.monotonic() - last_status_time > 1.0:
            print("waiting for serial data...")
            last_status_time = time.monotonic()

        header_indexes = [
            index
            for header in FRAME_HEADERS
            for index in [buffer.find(header)]
            if index >= 0
        ]
        header_index = min(header_indexes) if header_indexes else -1
        if header_index < 0:
            if len(buffer) > 1:
                if debug:
                    print(f"drop bytes before header: {bytes(buffer[:-1]).hex(' ')}")
                del buffer[:-1]
            continue

        if header_index > 0:
            del buffer[:header_index]

        if len(buffer) < FRAME_LEN:
            continue

        frame = bytes(buffer[:FRAME_LEN])
        if frame[-2:] != FRAME_TAIL:
            print(f"frame tail mismatch, resync: {frame.hex(' ')}")
            del buffer[0]
            continue
        del buffer[:FRAME_LEN]
        return frame


def read_one_sample(ser: serial.Serial, buffer: bytearray, command: bytes, request_mode: bool, debug=False):
    if request_mode:
        ser.write(command)
        if debug:
            print("tx command")
        time.sleep(0.002)
    frame = read_exact_frame(ser, buffer, debug=debug)
    return parse_frame(frame)


def tare_sensor(ser, buffer, command, request_mode, sample_count=100, debug=False):
    samples = []
    print(f"tare start: collecting {sample_count} samples, keep sensor unloaded and still...")
    for _ in range(sample_count):
        try:
            samples.append(read_one_sample(ser, buffer, command, request_mode, debug=debug))
        except ValueError as exc:
            print(exc)
    if not samples:
        raise RuntimeError("tare failed: no valid samples")
    bias = tuple(sum(sample[i] for sample in samples) / len(samples) for i in range(6))
    print(
        "tare bias kg/kg*m: "
        f"Fx={bias[0]:.4f}, Fy={bias[1]:.4f}, Fz={bias[2]:.4f}, "
        f"Mx={bias[3]:.6f}, My={bias[4]:.6f}, Mz={bias[5]:.6f}"
    )
    return bias


def load_bias(path: Path) -> tuple[float, float, float, float, float, float]:
    if not path.exists():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    bias = payload.get("bias_kg_kgm", [0.0] * 6)
    if len(bias) != 6:
        raise ValueError(f"Expected 6 bias values in {path}")
    return tuple(float(value) for value in bias)


def save_bias(path: Path, bias: tuple[float, float, float, float, float, float]) -> None:
    payload = {
        "bias_kg_kgm": [float(value) for value in bias],
        "updated_at_s": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
