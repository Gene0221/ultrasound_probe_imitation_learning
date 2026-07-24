from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
from typing import Any

import yaml


CONTROLLER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CONTROLLER_ROOT.parent


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the realtime libfranka rolling policy controller.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "infer.yaml"))
    parser.add_argument("--robot-ip", default=None, help="Override franka.robot_ip from config.")
    parser.add_argument("--binary", default=str(CONTROLLER_ROOT / "build" / "rolling_policy_controller"))
    parser.add_argument("--print-only", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def add_flag(command: list[str], name: str, value: Any) -> None:
    command.extend([name, str(value)])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config.get("runtime", {})
    motion = config.get("motion", {})
    limits = motion.get("limits", {})
    online_filter = motion.get("online_filter", {})
    franka = config.get("franka", {})
    robot_ip = args.robot_ip or franka.get("robot_ip")
    if not robot_ip:
        raise ValueError("Robot IP is required. Set franka.robot_ip in config or pass --robot-ip.")

    command = [str(Path(args.binary).resolve()), "--robot-ip", robot_ip]
    add_flag(command, "--host", runtime.get("host", "127.0.0.1"))
    add_flag(command, "--port", int(runtime.get("port", 50555)))
    add_flag(command, "--receive-timeout-ms", int(runtime.get("receive_timeout_ms", 150)))
    add_flag(command, "--action-horizon", int(motion.get("action_horizon", config.get("policy", {}).get("action_horizon", 20))))
    add_flag(command, "--max-step-translation", limits.get("max_step_translation_m", 0.003))
    add_flag(command, "--max-step-rotation", limits.get("max_step_rotation_rad", 0.05))
    add_flag(command, "--max-translation-speed", limits.get("max_translation_speed_mps", 0.2))
    add_flag(command, "--max-translation-acceleration", limits.get("max_translation_acceleration_mps2", 0.07))
    add_flag(command, "--max-rotation-speed", limits.get("max_rotation_speed_radps", 0.35))
    add_flag(command, "--max-rotation-acceleration", limits.get("max_rotation_acceleration_radps2", 0.5))
    add_flag(command, "--ramp-time", motion.get("ramp_time_s", 3.0))

    if not bool(online_filter.get("enabled", True)):
        command.append("--disable-filter")
    else:
        add_flag(command, "--filter-cutoff-hz", online_filter.get("cutoff_hz", 1.0))
    if not bool(online_filter.get("orientation_enabled", True)):
        command.append("--disable-orientation-filter")
    else:
        add_flag(command, "--orientation-filter-cutoff-hz", online_filter.get("orientation_cutoff_hz", 1.0))

    print(" ".join(shlex.quote(item) for item in command))
    if not args.print_only:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
