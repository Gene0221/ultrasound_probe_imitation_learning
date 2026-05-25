from __future__ import annotations

import argparse

from common import load_config, load_json, run_pair_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate extrinsics between two synchronized sensors.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config.")
    parser.add_argument("--session-name", required=True, help="Saved capture session name.")
    parser.add_argument("--pair-id", required=True, help="Pair ID defined in config, such as rgb_to_ir_left.")
    parser.add_argument("--source-calibration", required=True, help="Path to the source sensor calibration JSON.")
    parser.add_argument("--target-calibration", required=True, help="Path to the target sensor calibration JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_calibration = load_json(args.source_calibration)
    target_calibration = load_json(args.target_calibration)
    result = run_pair_calibration(config, args.session_name, args.pair_id, source_calibration, target_calibration)
    print(f"[DONE] Extrinsic calibration saved for '{args.pair_id}'.")
    print(f"[DONE] Stereo RMS error: {result['stereo_rms']:.6f}")


if __name__ == "__main__":
    main()
