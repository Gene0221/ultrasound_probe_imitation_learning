from __future__ import annotations

import argparse
from pathlib import Path

import serial

from kwr75b_common import load_config, print_ports, resolve_port, resolve_workspace_path, save_bias, tare_sensor


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero calibration for the KWR75B sensor and save bias values.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--port", default=None, help="serial port, for example /dev/ttyUSB0 or COM3")
    parser.add_argument("--serial-number", default=None, help="USB serial adapter serial number")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_ports:
        print_ports()
        return

    config = load_config(args.config)
    if args.port is not None:
        config.setdefault("serial", {})["port"] = args.port
    if args.serial_number is not None:
        config.setdefault("serial", {})["serial_number"] = args.serial_number
    if args.baudrate is not None:
        config.setdefault("serial", {})["baudrate"] = args.baudrate

    port = resolve_port(config)
    baudrate = int(config.get("serial", {}).get("baudrate", 460800))
    request_mode = bool(config.get("serial", {}).get("request_mode", True))
    debug = bool(config.get("serial", {}).get("debug", False))
    sample_count = int(config.get("calibration", {}).get("tare_samples", 100))
    bias_path = resolve_workspace_path(config.get("calibration", {}).get("bias_file", "config/zero_bias.json"))

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

    try:
        bias = tare_sensor(ser, buffer, command, request_mode, sample_count=sample_count, debug=debug)
        save_bias(bias_path, bias)
        print(f"[INFO] Saved zero calibration to {bias_path}")
    except KeyboardInterrupt:
        print("\n[INFO] Calibration interrupted by user.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
