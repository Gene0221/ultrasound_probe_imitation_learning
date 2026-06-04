from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import serial

from kwr75b_common import (
    PROJECT_ROOT,
    load_bias,
    load_config,
    print_ports,
    read_one_sample,
    read_keyboard_char,
    resolve_port,
    resolve_workspace_path,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"


def append_jsonl(handle: Any, timestamp_s: float, raw: tuple[float, ...], zeroed: tuple[float, ...]) -> None:
    fx, fy, fz, mx, my, mz = zeroed
    payload = {
        "host_timestamp_s": timestamp_s,
        "Fz": fz * 9.80665,
        "raw_Fz_kg": raw[2],
        "zeroed_Fz_kg": fz,
    }
    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    handle.flush()


def load_control_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"recording": True, "output_dir": None, "shutdown": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Control file must contain a JSON object: {path}")
    return payload


def read_kwr75b(
    port="COM3",
    baudrate=460800,
    request_mode=True,
    debug=False,
    bias=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    sampling_hz=50.0,
    output_root="output",
    file_name="force6d.jsonl",
    print_human_readable=True,
    control_file: Path | None = None,
):
    print(f"open {port}, baudrate={baudrate}, request_mode={request_mode}")

    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=8,
        stopbits=1,
        parity="N",
        timeout=0.1,
    )

    command = bytes([0x49, 0xAA, 0x0D, 0x0A])
    buffer = bytearray()
    ser.reset_input_buffer()
    print(f"send command: {command.hex(' ')}")
    print("keys: q=quit")

    output_dir = resolve_workspace_path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    period_s = 1.0 / max(float(sampling_hz), 1e-6)
    log_handle: Any | None = None
    active_output_dir: Path | None = None

    try:
        while True:
            control_state = load_control_state(control_file)
            if bool(control_state.get("shutdown", False)):
                print("shutdown requested")
                break

            recording = bool(control_state.get("recording", control_file is None))
            output_dir_value = control_state.get("output_dir")
            requested_output_dir = Path(output_dir_value).resolve() if output_dir_value else output_dir

            if recording and active_output_dir != requested_output_dir:
                if log_handle is not None:
                    log_handle.close()
                requested_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = requested_output_dir / file_name
                print(f"jsonl output: {output_path}")
                log_handle = output_path.open("w", encoding="utf-8")
                active_output_dir = requested_output_dir
            elif not recording and active_output_dir is not None:
                if log_handle is not None:
                    log_handle.close()
                    log_handle = None
                active_output_dir = None

            if control_file is None:
                key = read_keyboard_char()
                if key == "q":
                    print("quit")
                    break

            try:
                raw = read_one_sample(ser, buffer, command, request_mode, debug=debug)
            except ValueError as exc:
                print(exc)
                continue

            fx, fy, fz, mx, my, mz = tuple(raw[i] - bias[i] for i in range(6))
            timestamp_s = time.time()
            if recording and log_handle is not None:
                append_jsonl(log_handle, timestamp_s, raw, (fx, fy, fz, mx, my, mz))

            if print_human_readable:
                print(
                    f"raw kg/kg*m: "
                    f"Fx={raw[0]:.4f}, Fy={raw[1]:.4f}, Fz={raw[2]:.4f}, "
                    f"Mx={raw[3]:.6f}, My={raw[4]:.6f}, Mz={raw[5]:.6f}"
                )
                print(f"zeroed: Fx={fx:.4f} kg  Fy={fy:.4f} kg  Fz={fz:.4f} kg")
                print(f"zeroed: Mx={mx:.6f} kg*m  My={my:.6f} kg*m  Mz={mz:.6f} kg*m")
                print(f"zeroed: Fx={fx * 9.80665:.3f} N  Fy={fy * 9.80665:.3f} N  Fz={fz * 9.80665:.3f} N")
                print(
                    f"zeroed: Mx={mx * 9.80665:.4f} N*m  "
                    f"My: {my * 9.80665:.4f} N*m  "
                    f"Mz: {mz * 9.80665:.4f} N*m"
                )
                print("-" * 60)

            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted acquisition. Exiting cleanly.")
    finally:
        if log_handle is not None:
            log_handle.close()
        ser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read KWR75B 6D force/torque data, print live values, and save JSONL logs."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--port", default=None, help="serial port, for example /dev/ttyUSB0 or COM3")
    parser.add_argument("--serial-number", default=None, help="USB serial adapter serial number")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    parser.add_argument("--control-file", default=None, help="Optional JSON control file for start/pause/stop.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_ports:
        print_ports()
        raise SystemExit(0)

    config = load_config(args.config)
    if args.port is not None:
        config.setdefault("serial", {})["port"] = args.port
    if args.serial_number is not None:
        config.setdefault("serial", {})["serial_number"] = args.serial_number
    if args.baudrate is not None:
        config.setdefault("serial", {})["baudrate"] = args.baudrate

    bias_path = resolve_workspace_path(config.get("calibration", {}).get("bias_file", "config/zero_bias.json"))
    bias = load_bias(bias_path)
    print(f"[INFO] Loaded zero bias from {bias_path}")
    read_kwr75b(
        port=resolve_port(config),
        baudrate=int(config.get("serial", {}).get("baudrate", 460800)),
        request_mode=bool(config.get("serial", {}).get("request_mode", True)),
        debug=bool(config.get("serial", {}).get("debug", False)),
        bias=bias,
        sampling_hz=float(config.get("sampling_hz", 50.0)),
        output_root=str(config.get("output_root", "output")),
        file_name=str(config.get("file_name", "force6d.jsonl")),
        print_human_readable=bool(config.get("print_human_readable", True)),
        control_file=Path(args.control_file).resolve() if args.control_file else None,
    )
