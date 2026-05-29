from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paxini26d_mapping.common import ensure_dir, env_with_pythonpath, load_json, next_session_name, resolve_from, save_json, utc_now_iso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified calibration for Paxini and 6D force sensors.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.json"))
    return parser.parse_args()


def run_module(
    module_name: str,
    module_cfg: dict[str, Any],
    python_executable: str,
) -> dict[str, Any]:
    workspace_root = resolve_from(PROJECT_ROOT, module_cfg["workspace_root"])
    script_path = resolve_from(workspace_root, module_cfg["script_path"])
    config_path = resolve_from(workspace_root, module_cfg["base_config_path"])
    command = [python_executable, str(script_path), "--config", str(config_path)]

    print(f"[INFO] Starting calibration for {module_name}")
    print(f"[INFO] Command: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(workspace_root),
        env=env_with_pythonpath(SRC_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Calibration failed for {module_name} with return code {completed.returncode}")

    artifacts: list[dict[str, str]] = []
    for artifact_rel in module_cfg.get("artifact_paths", []):
        artifact_path = resolve_from(workspace_root, artifact_rel)
        if artifact_path.exists():
            artifacts.append({"source_path": str(artifact_path), "relative_path": str(artifact_rel)})
        else:
            print(f"[WARN] Expected calibration artifact not found for {module_name}: {artifact_path}")

    return {
        "workspace_root": str(workspace_root),
        "script_path": str(script_path),
        "config_path": str(config_path),
        "artifacts": artifacts,
    }


def archive_artifacts(
    module_name: str,
    artifacts: list[dict[str, str]],
    run_dir: Path,
) -> list[str]:
    archived_paths: list[str] = []
    module_dir = ensure_dir(run_dir / module_name)
    for artifact in artifacts:
        source_path = Path(artifact["source_path"])
        destination_path = module_dir / Path(artifact["relative_path"]).name
        ensure_dir(destination_path.parent)
        shutil.copy2(source_path, destination_path)
        archived_paths.append(str(destination_path))
    return archived_paths


def main() -> None:
    args = parse_args()
    config = load_json(resolve_from(PROJECT_ROOT, args.config))
    calibration_cfg = config["calibration"]
    calibration_root = ensure_dir(resolve_from(PROJECT_ROOT, config["paths"]["calibration_root"]))
    run_name = next_session_name(calibration_root, calibration_cfg["run_prefix"])
    run_dir = ensure_dir(calibration_root / run_name)

    run_meta: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "started_at_utc": utc_now_iso(),
        "status": "running",
        "modules": {},
    }
    save_json(run_dir / "calibration_run.json", run_meta)

    try:
        for module_name, module_cfg in calibration_cfg["modules"].items():
            module_result = run_module(
                module_name=module_name,
                module_cfg=module_cfg,
                python_executable=str(calibration_cfg["python_executable"]),
            )
            archived_paths = archive_artifacts(module_name, module_result["artifacts"], run_dir)
            module_result["archived_paths"] = archived_paths
            run_meta["modules"][module_name] = module_result

        run_meta["status"] = "completed"
        print(f"[INFO] Archived calibration bundle at {run_dir}")
    except Exception as exc:
        run_meta["status"] = "failed"
        run_meta["error"] = str(exc)
        raise
    finally:
        run_meta["finished_at_utc"] = utc_now_iso()
        save_json(run_dir / "calibration_run.json", run_meta)


if __name__ == "__main__":
    main()
