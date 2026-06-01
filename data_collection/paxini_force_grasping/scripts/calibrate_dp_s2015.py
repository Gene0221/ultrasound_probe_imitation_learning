from __future__ import annotations

import argparse

from paxini_common import DEFAULT_CONFIG_PATH, configure_dp_module, load_config, load_dp_module, print_ports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero calibration for dual DP-S2015 sensors.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--port", default=None, help="serial port, for example /dev/ttyUSB0 or COM6")
    parser.add_argument("--serial-number", default=None, help="USB serial adapter serial number")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_ports:
        print_ports()
        raise SystemExit(0)

    config = load_config(args.config)
    if args.port is not None:
        config.setdefault("serial", {})["port"] = args.port
    if args.serial_number is not None:
        config.setdefault("serial", {})["serial_number"] = args.serial_number
    if args.baudrate is not None:
        config.setdefault("serial", {})["baudrate"] = args.baudrate

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
