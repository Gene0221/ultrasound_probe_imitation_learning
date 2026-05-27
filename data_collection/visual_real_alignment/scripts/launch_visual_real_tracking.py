from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DATA_COLLECTION_ROOT = WORKSPACE_ROOT.parent
REAL_WORKSPACE_ROOT = DATA_COLLECTION_ROOT / "real_pose_tracking"
VISUAL_WORKSPACE_ROOT = DATA_COLLECTION_ROOT / "visual_pose_tracking"
DEFAULT_REAL_LAUNCH_SCRIPT = REAL_WORKSPACE_ROOT / "launch.sh"
DEFAULT_REAL_CONFIG = REAL_WORKSPACE_ROOT / "config" / "default.yaml"
DEFAULT_VISUAL_CONFIG = VISUAL_WORKSPACE_ROOT / "config" / "apriltag_tracking.yaml"
DEFAULT_VISUAL_SCRIPT = VISUAL_WORKSPACE_ROOT / "scripts" / "track_apriltag_pose_deltas.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch visual and real pose tracking together.")
    parser.add_argument(
        "--real-launch-script",
        type=Path,
        default=DEFAULT_REAL_LAUNCH_SCRIPT,
        help="Path to the real pose tracking launch script.",
    )
    parser.add_argument(
        "--visual-script",
        type=Path,
        default=DEFAULT_VISUAL_SCRIPT,
        help="Path to the visual pose tracking Python entrypoint.",
    )
    parser.add_argument(
        "--visual-config",
        type=Path,
        default=DEFAULT_VISUAL_CONFIG,
        help="Path to the visual pose tracking config YAML.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch the visual tracker.",
    )
    parser.add_argument(
        "--visual-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to the visual tracker after '--visual-extra-args'.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {description}: {path}")
    return path


def stream_output(prefix: str, pipe, stop_event: threading.Event) -> None:
    try:
        assert pipe is not None
        for raw_line in iter(pipe.readline, ""):
            if not raw_line:
                break
            print(f"[{prefix}] {raw_line.rstrip()}")
            if stop_event.is_set():
                break
    finally:
        if pipe is not None:
            pipe.close()


def terminate_process(process: subprocess.Popen[str], name: str, timeout_s: float = 5.0) -> None:
    if process.poll() is not None:
        return

    print(f"[INFO] Stopping {name}...")
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
    except Exception:
        process.terminate()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)

    process.kill()


def main() -> None:
    args = parse_args()
    real_launch_script = require_file(args.real_launch_script, "real tracking launch script")
    visual_script = require_file(args.visual_script, "visual tracking script")
    visual_config = require_file(args.visual_config, "visual tracking config")
    python_executable = require_file(args.python_executable, "Python executable")

    real_command = ["bash", str(real_launch_script)]
    visual_command = [str(python_executable), str(visual_script), "--config", str(visual_config), *args.visual_extra_args]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    print(f"[INFO] Real command: {' '.join(real_command)}")
    print(f"[INFO] Visual command: {' '.join(visual_command)}")

    stop_event = threading.Event()
    real_process: subprocess.Popen[str] | None = None
    visual_process: subprocess.Popen[str] | None = None
    stream_threads: list[threading.Thread] = []

    try:
        real_process = subprocess.Popen(
            real_command,
            cwd=str(REAL_WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        visual_process = subprocess.Popen(
            visual_command,
            cwd=str(VISUAL_WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )

        stream_threads = [
            threading.Thread(target=stream_output, args=("REAL", real_process.stdout, stop_event), daemon=True),
            threading.Thread(target=stream_output, args=("VISUAL", visual_process.stdout, stop_event), daemon=True),
        ]
        for thread in stream_threads:
            thread.start()

        while True:
            real_code = real_process.poll()
            visual_code = visual_process.poll()
            if real_code is not None or visual_code is not None:
                break
            time.sleep(0.2)

        if real_process.poll() is not None:
            print(f"[WARN] Real process exited with code {real_process.returncode}.")
        if visual_process.poll() is not None:
            print(f"[WARN] Visual process exited with code {visual_process.returncode}.")

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received.")
    finally:
        stop_event.set()
        if visual_process is not None:
            terminate_process(visual_process, "visual tracker")
        if real_process is not None:
            terminate_process(real_process, "real tracker")
        for thread in stream_threads:
            thread.join(timeout=1.0)

    if real_process is not None and real_process.returncode not in (None, 0):
        raise SystemExit(real_process.returncode)
    if visual_process is not None and visual_process.returncode not in (None, 0):
        raise SystemExit(visual_process.returncode)


if __name__ == "__main__":
    main()
