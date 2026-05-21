from __future__ import annotations

import argparse

from common import load_config, run_mono_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate one sensor from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config.")
    parser.add_argument("--session-name", required=True, help="Saved capture session name.")
    parser.add_argument("--sensor-id", required=True, help="Sensor ID defined in config, such as rgb or ir_left.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_mono_calibration(config, args.session_name, args.sensor_id)
    print(f"[DONE] Calibration saved for sensor '{args.sensor_id}'.")
    print(f"[DONE] RMS error: {result['rms']:.6f}")
    print(f"[DONE] Mean reprojection error: {result['mean_reprojection_error']:.6f}")


if __name__ == "__main__":
    main()
