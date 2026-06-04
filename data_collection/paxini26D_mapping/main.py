from __future__ import annotations

import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
HOSPITAL_SRC = WORKSPACE_ROOT.parent / "hospital_data_collection" / "src"
if str(HOSPITAL_SRC) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_SRC))

from hospital_data_collection.launcher import CollectionLauncher


def run_collection() -> None:
    launcher = CollectionLauncher.from_config_path(WORKSPACE_ROOT / "config" / "default.yaml")
    launcher.run_interactive()


def run_prepare_dataset() -> None:
    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "prepare_dataset.py"),
        "--config",
        str(WORKSPACE_ROOT / "config" / "default.yaml"),
    ]
    completed = subprocess.run(command, cwd=str(WORKSPACE_ROOT), check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"prepare_dataset.py failed with return code {completed.returncode}")


def run_train_model() -> None:
    command = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "train_model.py"),
        "--config",
        str(WORKSPACE_ROOT / "config" / "default.yaml"),
    ]
    completed = subprocess.run(command, cwd=str(WORKSPACE_ROOT), check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"train_model.py failed with return code {completed.returncode}")


def main() -> None:
    run_collection()
    print("[INFO] Collection finished. Preparing aligned dataset...")
    run_prepare_dataset()
    print("[INFO] Dataset prepared. Training mapping model...")
    run_train_model()
    print("[INFO] Collect-train pipeline completed.")


if __name__ == "__main__":
    main()
