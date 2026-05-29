from __future__ import annotations

import argparse

from paxini_common import DEFAULT_CONFIG_PATH, configure_dp_module, load_config, load_dp_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero calibration for dual DP-S2015 sensors.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    dp = load_dp_module()
    calibration_path = configure_dp_module(dp, config)
    board = dp.HandBoard(port=dp.PORT)

    try:
        board.connect()
        board.get_version()
        board.get_sensor_types()
        board.get_data_points()

        print("Keep both DP-S2015 sensors unloaded and still.")
        input("Press Enter to collect zero calibration...")

        calibration = board.create_zero_calibration(samples=dp.ZERO_CHECK_SAMPLES)
        dp.print_calibration_summary(calibration)

        choice = input("Save this calibration? Enter y to save, other key to discard: ").strip().lower()
        if choice in ("y", "yes"):
            dp.save_calibration(calibration, calibration_path)
        else:
            print("calibration discarded")
    except KeyboardInterrupt:
        print("\nstop requested by user")
    finally:
        board.close()
