from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import get_sensor_ids, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate all enabled sensors from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config.")
    parser.add_argument("--session-name", required=True, help="Saved capture session name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sensor_ids = get_sensor_ids(config)
    script_path = Path(__file__).resolve().parent / "calibrate_camera.py"

    for sensor_id in sensor_ids:
        print(f"\n===== Calibrating {sensor_id} =====")
        result = subprocess.run(
            [sys.executable, str(script_path), "--config", args.config, "--session-name", args.session_name, "--sensor-id", sensor_id],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"Calibration failed for sensor '{sensor_id}'.")


if __name__ == "__main__":
    main()
