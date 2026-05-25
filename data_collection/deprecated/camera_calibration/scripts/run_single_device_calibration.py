from __future__ import annotations

import argparse

from common import (
    get_pair_paths,
    get_sensor_paths,
    load_config,
    run_mono_calibration,
    run_pair_calibration,
    summarize_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full calibration for one D435i capture session.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config.")
    parser.add_argument("--session-name", required=True, help="Saved capture session name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    session_name = args.session_name

    mono_results: dict[str, dict] = {}
    for sensor_id in ("rgb", "ir_left", "ir_right"):
        print(f"\n===== Mono calibration: {sensor_id} =====")
        mono_results[sensor_id] = run_mono_calibration(config, session_name, sensor_id)
        sensor_paths = get_sensor_paths(config, session_name, sensor_id)
        print(f"[DONE] Saved: {sensor_paths.calibration_file}")

    pair_results: dict[str, dict] = {}
    print("\n===== Stereo calibration: ir_left <-> ir_right =====")
    pair_results["ir_stereo"] = run_pair_calibration(
        config,
        session_name,
        "ir_stereo",
        mono_results["ir_left"],
        mono_results["ir_right"],
    )
    print(f"[DONE] Saved: {get_pair_paths(config, session_name, 'ir_stereo').calibration_file}")

    print("\n===== Extrinsic calibration: rgb <-> ir_left =====")
    pair_results["rgb_to_ir_left"] = run_pair_calibration(
        config,
        session_name,
        "rgb_to_ir_left",
        mono_results["rgb"],
        mono_results["ir_left"],
    )
    print(f"[DONE] Saved: {get_pair_paths(config, session_name, 'rgb_to_ir_left').calibration_file}")

    bundle = summarize_bundle(config, session_name, mono_results, pair_results)
    print("\n===== Summary =====")
    print(f"[DONE] Session: {session_name}")
    print(f"[DONE] Bundle sensors: {', '.join(bundle['sensors'].keys())}")
    print(f"[DONE] Bundle pairs: {', '.join(bundle['pairs'].keys())}")


if __name__ == "__main__":
    main()
