from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from paxini_common import DEFAULT_CONFIG_PATH, configure_dp_module, load_config, load_dp_module, resolve_workspace_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read dual DP-S2015 sensors and save left/right JSONL logs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def write_record(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dp = load_dp_module()
    calibration_path = configure_dp_module(dp, config)

    output_root = resolve_workspace_path(config.get("output_root", "output"))
    output_root.mkdir(parents=True, exist_ok=True)
    left_path = output_root / str(config.get("left_file_name", "left_sensor.jsonl"))
    right_path = output_root / str(config.get("right_file_name", "right_sensor.jsonl"))
    print_human_readable = bool(config.get("print_human_readable", True))

    board = dp.HandBoard(port=dp.PORT)
    stop_requested = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[INFO] Paxini logger received signal {signum}; stopping...", file=sys.stderr)

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        board.connect()
        board.get_version()
        board.get_sensor_types()
        board.get_data_points()
        calibration = dp.load_calibration(calibration_path)
        if calibration:
            dp.print_calibration_summary(calibration)
        else:
            print("no calibration loaded; output will use raw sensor values")

        with left_path.open("w", encoding="utf-8") as left_handle, right_path.open("w", encoding="utf-8") as right_handle:
            board.start_stream()
            try:
                while not stop_requested:
                    sensors = board.read_stream_sensors(timeout=1.0)
                    if not sensors:
                        print("stream timeout")
                        continue
                    sensors = dp.apply_calibration(sensors, calibration)
                    timestamp_s = time.time()
                    for sensor in sensors:
                        payload = {
                            "host_timestamp_s": timestamp_s,
                            "sensor_index": sensor["sensor_index"],
                            "label": sensor["label"],
                            "point_count": sensor["point_count"],
                            "values": [
                                float(sensor["total_force"]["Fx"]),
                                float(sensor["total_force"]["Fy"]),
                                float(sensor["total_force"]["Fz"]),
                            ],
                            "total_force": sensor["total_force"],
                            "points": sensor["points"],
                        }
                        if sensor["sensor_index"] == 0:
                            write_record(left_handle, payload)
                        elif sensor["sensor_index"] == 1:
                            write_record(right_handle, payload)

                    if print_human_readable:
                        parts = []
                        for sensor in sensors:
                            total = sensor["total_force"]
                            parts.append(
                                f"{sensor['label']}: Fx={total['Fx']:+.1f}N Fy={total['Fy']:+.1f}N Fz={total['Fz']:.1f}N"
                            )
                        print(" | ".join(parts))
            finally:
                board.stop_stream()
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted acquisition. Exiting cleanly.")
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        board.close()


if __name__ == "__main__":
    main()
