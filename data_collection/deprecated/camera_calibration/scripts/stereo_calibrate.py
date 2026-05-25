from __future__ import annotations

import argparse

from common import load_config, load_json, run_pair_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate one synchronized sensor pair.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config.")
    parser.add_argument("--session-name", required=True, help="Saved capture session name.")
    parser.add_argument("--pair-id", required=True, help="Pair ID defined in config, such as ir_stereo.")
    parser.add_argument("--left-calibration", required=True, help="Path to the left sensor calibration JSON.")
    parser.add_argument("--right-calibration", required=True, help="Path to the right sensor calibration JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    left_calibration = load_json(args.left_calibration)
    right_calibration = load_json(args.right_calibration)
    result = run_pair_calibration(config, args.session_name, args.pair_id, left_calibration, right_calibration)
    print(f"[DONE] Pair calibration saved for '{args.pair_id}'.")
    print(f"[DONE] Stereo RMS error: {result['stereo_rms']:.6f}")


if __name__ == "__main__":
    main()
