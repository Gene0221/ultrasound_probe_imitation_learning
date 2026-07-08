from __future__ import annotations

import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
RUNTIME_SRC = WORKSPACE_ROOT.parent / "collection_runtime" / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from collection_runtime.launcher import CollectionLauncher


def prompt_experiment_name() -> str:
    while True:
        experiment_name = input("Enter experiment name: ").strip()
        if experiment_name:
            safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in experiment_name)
            if safe_name != experiment_name:
                print(f"[INFO] Using sanitized experiment name: {safe_name}")
            return safe_name
        print("[WARN] Experiment name cannot be empty.")


def run_collection() -> None:
    launcher = CollectionLauncher.from_config_path(WORKSPACE_ROOT / "config" / "default.yaml")
    launcher.run_interactive()


def run_prepare_dataset(experiment_name: str) -> None:
    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "prepare_dataset.py"),
        "--config",
        str(WORKSPACE_ROOT / "config" / "default.yaml"),
        "--experiment-name",
        experiment_name,
    ]
    completed = subprocess.run(command, cwd=str(WORKSPACE_ROOT), check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"prepare_dataset.py failed with return code {completed.returncode}")


def run_train_model(experiment_name: str) -> None:
    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "train_model.py"),
        "--config",
        str(WORKSPACE_ROOT / "config" / "default.yaml"),
        "--experiment-name",
        experiment_name,
    ]
    completed = subprocess.run(command, cwd=str(WORKSPACE_ROOT), check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"train_model.py failed with return code {completed.returncode}")


def main() -> None:
    experiment_name = prompt_experiment_name()
    run_collection()
    print("[INFO] Collection finished. Preparing aligned dataset...")
    run_prepare_dataset(experiment_name)
    print("[INFO] Dataset prepared. Training mapping model...")
    run_train_model(experiment_name)
    print("[INFO] Collect-train pipeline completed.")


if __name__ == "__main__":
    main()
