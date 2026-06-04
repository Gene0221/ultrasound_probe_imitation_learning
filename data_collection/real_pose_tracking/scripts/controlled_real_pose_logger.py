from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control real pose tracking with a session control file.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--control-file", required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def load_control_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"recording": False, "output_dir": None, "shutdown": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Control file must contain a JSON object: {path}")
    return payload


def find_binary() -> Path:
    candidates = [
        PROJECT_ROOT / "build" / "read_franka_ee_pose",
        PROJECT_ROOT / "build" / "Release" / "read_franka_ee_pose",
        PROJECT_ROOT / "src" / "read_franka_ee_pose",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Cannot find read_franka_ee_pose binary under build/, build/Release/, or src/.")


def terminate_process(process: subprocess.Popen[str] | None, timeout_s: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.config).resolve()
    control_file = Path(args.control_file).resolve()
    binary_path = find_binary()

    process: subprocess.Popen[str] | None = None
    active_output_dir: Path | None = None

    try:
        while True:
            control_state = load_control_state(control_file)
            if bool(control_state.get("shutdown", False)):
                break

            recording = bool(control_state.get("recording", False))
            output_dir_value = control_state.get("output_dir")
            requested_output_dir = Path(output_dir_value).resolve() if output_dir_value else None

            if recording and requested_output_dir is not None and active_output_dir != requested_output_dir:
                terminate_process(process)
                process = None
                runtime_config = load_yaml(base_config_path)
                runtime_config.setdefault("output", {})["output_root"] = str(requested_output_dir)
                runtime_config["output"]["jsonl_file_name"] = "franka_ee_pose_deltas.jsonl"
                runtime_config["output"]["summary_file_name"] = "summary.json"
                runtime_config_path = requested_output_dir / "real_pose_runtime.yaml"
                save_yaml(runtime_config_path, runtime_config)
                requested_output_dir.mkdir(parents=True, exist_ok=True)
                process = subprocess.Popen(
                    [str(binary_path), str(runtime_config_path)],
                    cwd=str(PROJECT_ROOT),
                    text=True,
                )
                active_output_dir = requested_output_dir
            elif not recording and process is not None:
                terminate_process(process)
                process = None
                active_output_dir = None

            time.sleep(0.2)
    finally:
        terminate_process(process)


if __name__ == "__main__":
    main()
