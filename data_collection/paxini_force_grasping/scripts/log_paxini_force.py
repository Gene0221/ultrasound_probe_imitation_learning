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
    parser.add_argument("--control-file", default=None, help="Optional JSON control file for start/pause/stop.")
    return parser.parse_args()


def write_record(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    handle.flush()


def load_control_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"recording": True, "output_dir": None, "shutdown": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Control file must contain a JSON object: {path}")
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dp = load_dp_module()
    calibration_path = configure_dp_module(dp, config)

    output_root = resolve_workspace_path(config.get("output_root", "output"))
    print_human_readable = bool(config.get("print_human_readable", True))
    control_file = Path(args.control_file).resolve() if args.control_file else None
    controlled_mode = control_file is not None
    if not controlled_mode:
        output_root.mkdir(parents=True, exist_ok=True)

    board = dp.HandBoard(port=dp.PORT)
    stop_requested = False
    left_handle: Any | None = None
    right_handle: Any | None = None
    active_output_dir: Path | None = None

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

        board.start_stream()
        try:
            while not stop_requested:
                control_state = load_control_state(control_file)
                if bool(control_state.get("shutdown", False)):
                    stop_requested = True
                    break

                recording = bool(control_state.get("recording", control_file is None))
                output_dir_value = control_state.get("output_dir")
                requested_output_dir = Path(output_dir_value).resolve() if output_dir_value else output_root
                if controlled_mode and output_dir_value is None:
                    requested_output_dir = None

                if recording and requested_output_dir is not None and active_output_dir != requested_output_dir:
                    if left_handle is not None:
                        left_handle.close()
                    if right_handle is not None:
                        right_handle.close()
                    requested_output_dir.mkdir(parents=True, exist_ok=True)
                    left_path = requested_output_dir / str(config.get("left_file_name", "left_sensor.jsonl"))
                    right_path = requested_output_dir / str(config.get("right_file_name", "right_sensor.jsonl"))
                    left_handle = left_path.open("w", encoding="utf-8")
                    right_handle = right_path.open("w", encoding="utf-8")
                    active_output_dir = requested_output_dir
                elif (not recording or requested_output_dir is None) and active_output_dir is not None:
                    if left_handle is not None:
                        left_handle.close()
                        left_handle = None
                    if right_handle is not None:
                        right_handle.close()
                        right_handle = None
                    active_output_dir = None

                sensors = board.read_stream_sensors(timeout=1.0)
                if not sensors:
                    if recording:
                        print("stream timeout")
                    continue
                sensors = dp.apply_calibration(sensors, calibration)
                timestamp_s = time.time()
                if recording and left_handle is not None and right_handle is not None:
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

                if recording and print_human_readable:
                    parts = []
                    for sensor in sensors:
                        total = sensor["total_force"]
                        parts.append(
                            f"{sensor['label']}: Fx={total['Fx']:+.1f}N Fy={total['Fy']:+.1f}N Fz={total['Fz']:.1f}N"
                        )
                    print(" | ".join(parts))
        finally:
            if left_handle is not None:
                left_handle.close()
            if right_handle is not None:
                right_handle.close()
            board.stop_stream()
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted acquisition. Exiting cleanly.")
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        board.close()


if __name__ == "__main__":
    main()
