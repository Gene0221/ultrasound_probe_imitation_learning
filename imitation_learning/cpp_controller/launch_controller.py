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


def active_policy_config(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("policy", {})
    if not isinstance(policy, dict):
        return {}
    policy_type = str(policy.get("type", "act")).lower()
    typed = policy.get(policy_type)
    if isinstance(typed, dict):
        merged = {key: value for key, value in policy.items() if key not in {"act", "diffusion"}}
        merged.update(typed)
        return merged
    return policy


def policy_type(config: dict[str, Any]) -> str:
    policy = config.get("policy", {})
    if not isinstance(policy, dict):
        return "act"
    return str(policy.get("type", "act")).lower()


def active_motion_config(config: dict[str, Any], selected_policy_type: str) -> dict[str, Any]:
    motion = config.get("motion", {})
    if not isinstance(motion, dict):
        return {}
    typed = motion.get(selected_policy_type)
    if isinstance(typed, dict):
        merged = {key: value for key, value in motion.items() if key not in {"act", "diffusion"}}
        merged.update(typed)
        return merged
    return motion


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config.get("runtime", {})
    selected_policy_type = policy_type(config)
    motion = active_motion_config(config, selected_policy_type)
    limits = motion.get("limits", {})
    online_filter = motion.get("online_filter", {})
    calibration = config.get("calibration", {})
    orientation_calibration = calibration.get("orientation", {})
    force_calibration = calibration.get("force", {})
    policy_cfg = active_policy_config(config)
    franka = config.get("franka", {})
    robot_ip = args.robot_ip or franka.get("robot_ip")
    if not robot_ip:
        raise ValueError("Robot IP is required. Set franka.robot_ip in config or pass --robot-ip.")

    command = [str(Path(args.binary).resolve()), "--robot-ip", robot_ip]
    add_flag(command, "--host", runtime.get("host", "127.0.0.1"))
    add_flag(command, "--port", int(runtime.get("port", 50555)))
    add_flag(command, "--action-horizon", int(motion.get("action_horizon", policy_cfg.get("action_horizon", 20))))
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

    orientation_enabled = bool(orientation_calibration.get("enabled", False))
    force_enabled = bool(force_calibration.get("enabled", False))
    if orientation_enabled or force_enabled:
        command.append("--enable-calibration")
    if not orientation_enabled:
        command.append("--disable-calibration-orientation")
    if not force_enabled:
        command.append("--disable-calibration-force")
    add_flag(command, "--calibration-interval-inferences", calibration.get("inferences_per_cycle", 3))
    add_flag(command, "--calibration-probe-z-offset", orientation_calibration.get("probe_z_offset_m", 0.0))
    add_flag(command, "--calibration-force-tolerance", force_calibration.get("tolerance_N", 0.5))
    ee_z = force_calibration.get("ee_z", {})
    add_flag(command, "--calibration-z-gain", ee_z.get("gain_m_per_N", 0.0002))
    add_flag(command, "--calibration-z-sign", ee_z.get("sign", 1.0))
    add_flag(command, "--calibration-max-z-step", ee_z.get("max_step_m", 0.0005))
    add_flag(command, "--calibration-max-total-z", ee_z.get("max_total_correction_m", 0.01))
    add_flag(command, "--calibration-z-settle-tolerance", ee_z.get("position_settle_tolerance_m", 0.0001))
    add_flag(command, "--calibration-z-settle-velocity", ee_z.get("position_settle_velocity_mps", 0.001))
    add_flag(command, "--calibration-orientation-tolerance", orientation_calibration.get("tolerance_rad", 0.01))
    add_flag(command, "--calibration-force-settle-cycles", calibration.get("settle_cycles", 3))
    add_flag(command, "--calibration-force-sample-hz", force_calibration.get("sample_hz", 30.0))

    print(" ".join(shlex.quote(item) for item in command))
    if not args.print_only:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
