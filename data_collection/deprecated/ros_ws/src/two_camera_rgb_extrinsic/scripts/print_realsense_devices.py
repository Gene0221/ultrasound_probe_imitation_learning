#!/usr/bin/env python3
from __future__ import annotations

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyrealsense2 is required to list RealSense devices. "
        "Install it first and then rerun this script."
    ) from exc


def main() -> None:
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    print("Detected RealSense devices:")
    for index, device in enumerate(devices, start=1):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        print(f"[{index}] name={name}")
        print(f"    serial_no={serial}")
        print(f"    firmware={firmware}")


if __name__ == "__main__":
    main()
