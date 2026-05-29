from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paxini26d_mapping.common import (  # noqa: E402
    ensure_dir,
    env_with_pythonpath,
    load_json,
    next_session_name,
    resolve_from,
    save_json,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch and archive one multimodal collection session.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.json"))
    return parser.parse_args()


def load_any_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected config object in {path}")
    return payload


def save_any_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def prepare_module_config(
    module_name: str,
    module_cfg: dict[str, Any],
    session_dir: Path,
    runtime_cfg_dir: Path,
) -> tuple[Path, Path]:
    workspace_root = resolve_from(PROJECT_ROOT, module_cfg["workspace_root"])
    base_config_path = resolve_from(workspace_root, module_cfg["base_config_path"])
    output_dir = ensure_dir(session_dir / module_cfg["output_subdir"])
    runtime_config = load_any_config(base_config_path)

    if module_name == "imu":
        output_cfg = runtime_config.setdefault("output", {})
        output_cfg["output_root"] = str(output_dir)
        output_cfg["jsonl_file_name"] = "imu_pitch_roll.jsonl"
        output_cfg["summary_file_name"] = "summary.json"
    elif module_name == "paxini":
        runtime_config["output_root"] = str(output_dir)
        runtime_config["left_file_name"] = "left_sensor.jsonl"
        runtime_config["right_file_name"] = "right_sensor.jsonl"
        if "print_human_readable" in module_cfg:
            runtime_config["print_human_readable"] = bool(module_cfg["print_human_readable"])
    elif module_name == "force6d":
        runtime_config["output_root"] = str(output_dir)
        runtime_config["file_name"] = "force6d.jsonl"
        if "print_human_readable" in module_cfg:
            runtime_config["print_human_readable"] = bool(module_cfg["print_human_readable"])
    else:
        raise KeyError(f"Unknown module: {module_name}")

    runtime_config_path = runtime_cfg_dir / f"{module_name}_runtime{base_config_path.suffix}"
    save_any_config(runtime_config_path, runtime_config)
    return workspace_root, runtime_config_path


def terminate_process(process: subprocess.Popen[Any], grace_s: float) -> dict[str, Any]:
    status: dict[str, Any] = {"pid": process.pid}
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_s)
            status["returncode"] = process.returncode
            status["terminated"] = True
            return status
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=grace_s)
            status["returncode"] = process.returncode
            status["terminated"] = False
            status["killed"] = True
            return status

    status["returncode"] = process.returncode
    status["terminated"] = False
    return status


def main() -> None:
    args = parse_args()
    config = load_json(resolve_from(PROJECT_ROOT, args.config))
    collect_cfg = config["collect"]
    paths_cfg = config["paths"]

    sessions_root = ensure_dir(resolve_from(PROJECT_ROOT, paths_cfg["sessions_root"]))
    session_name = next_session_name(sessions_root, collect_cfg["session_prefix"])
    session_dir = ensure_dir(sessions_root / session_name)
    metadata_dir = ensure_dir(session_dir / "metadata")
    runtime_cfg_dir = ensure_dir(metadata_dir / "runtime_configs")

    process_specs: dict[str, dict[str, Any]] = {}
    for module_name, module_cfg in collect_cfg["modules"].items():
        workspace_root, runtime_config_path = prepare_module_config(module_name, module_cfg, session_dir, runtime_cfg_dir)
        script_path = resolve_from(workspace_root, module_cfg["script_path"])
        process_specs[module_name] = {
            "workspace_root": workspace_root,
            "script_path": script_path,
            "runtime_config_path": runtime_config_path,
        }

    session_meta = {
        "session_name": session_name,
        "session_dir": str(session_dir),
        "started_at_utc": utc_now_iso(),
        "status": "running",
        "modules": {
            name: {
                "workspace_root": str(spec["workspace_root"]),
                "script_path": str(spec["script_path"]),
                "runtime_config_path": str(spec["runtime_config_path"]),
            }
            for name, spec in process_specs.items()
        },
    }
    save_json(metadata_dir / "session.json", session_meta)

    processes: dict[str, subprocess.Popen[Any]] = {}
    try:
        for module_name, spec in process_specs.items():
            command = [
                str(config["collect"]["python_executable"]),
                str(spec["script_path"]),
                "--config",
                str(spec["runtime_config_path"]),
            ]
            processes[module_name] = subprocess.Popen(
                command,
                cwd=str(spec["workspace_root"]),
                env=env_with_pythonpath(SRC_ROOT),
            )
            print(f"[INFO] Started {module_name}: pid={processes[module_name].pid}")

        print("[INFO] Collection is running. Press ENTER to stop all modules.")
        try:
            input()
        except KeyboardInterrupt:
            print("[INFO] Keyboard interrupt received; stopping all modules.")
    finally:
        process_results = {
            name: terminate_process(process, float(collect_cfg["termination_grace_s"]))
            for name, process in processes.items()
        }
        session_meta["finished_at_utc"] = utc_now_iso()
        session_meta["status"] = "completed"
        session_meta["process_results"] = process_results
        save_json(metadata_dir / "session.json", session_meta)
        time.sleep(0.2)
        print(f"[INFO] Archived session at {session_dir}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
