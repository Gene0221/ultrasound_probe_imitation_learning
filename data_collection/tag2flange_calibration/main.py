from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_COLLECTION_ROOT = WORKSPACE_ROOT.parent
VISUAL_WORKSPACE_ROOT = DATA_COLLECTION_ROOT / "visual_pose_tracking"
REAL_WORKSPACE_ROOT = DATA_COLLECTION_ROOT / "real_pose_tracking"
VISUAL_SCRIPT = VISUAL_WORKSPACE_ROOT / "scripts" / "track_apriltag_pose_deltas.py"
REAL_SCRIPT = REAL_WORKSPACE_ROOT / "scripts" / "controlled_real_pose_logger.py"
SOLVER_SCRIPT = WORKSPACE_ROOT / "scripts" / "solve_tag2flange_calibration.py"


def read_single_key() -> str:
    if sys.platform.startswith("win"):
        import msvcrt

        while True:
            key = msvcrt.getwch()
            if key in {"\x00", "\xe0"}:
                msvcrt.getwch()
                continue
            if key == "\r":
                return "ENTER"
            if key == "\x03":
                raise KeyboardInterrupt
            return key.lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1)
            if key == "\x1b":
                next_char = sys.stdin.read(1)
                if next_char == "[":
                    sys.stdin.read(1)
                    continue
                continue
            if key in {"\r", "\n"}:
                return "ENTER"
            if key == "\x03":
                raise KeyboardInterrupt
            return key.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def write_control(path: Path, recording: bool, output_dir: Path | None, shutdown: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recording": recording,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "shutdown": shutdown,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def wait_for_quit() -> None:
    print("[INFO] Tag2flange collection is running. Press q to stop and solve calibration.")
    while True:
        user_input = read_single_key()
        if user_input == "q":
            print()
            return


def prompt_experiment_id() -> str:
    while True:
        experiment_id = input("Enter experiment id: ").strip()
        if experiment_id:
            safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in experiment_id)
            if safe_id != experiment_id:
                print(f"[INFO] Using sanitized experiment id: {safe_id}")
            return safe_id
        print("[WARN] Experiment id cannot be empty.")


def solve_calibration(output_root: Path, visual_output: Path, real_output: Path, experiment_id: str) -> None:
    report_path = output_root / "tag2flange_calibration_report.json"
    bundle_path = output_root / "tag2flange_calibration_data.npz"
    command = [
        sys.executable,
        str(SOLVER_SCRIPT),
        "--experiment-id",
        experiment_id,
        "--visual-log",
        str(visual_output / "tag_pose_deltas.jsonl"),
        "--real-log",
        str(real_output / "franka_ee_pose_deltas.jsonl"),
        "--output-json",
        str(report_path),
        "--output-npz",
        str(bundle_path),
    ]
    print("[INFO] Solving tag-to-flange calibration...")
    completed = subprocess.run(command, cwd=str(WORKSPACE_ROOT), check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Calibration solver failed with return code {completed.returncode}")
    print(f"[INFO] Calibration report written to {report_path}")
    print(f"[INFO] Calibration bundle written to {bundle_path}")


def main() -> None:
    experiment_id = prompt_experiment_id()
    output_root = WORKSPACE_ROOT / "output" / f"experiment_{experiment_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    visual_output = output_root / "visual_pose"
    real_output = output_root / "real_pose"
    control_root = output_root / "controls"
    visual_control = control_root / "visual_control.json"
    real_control = control_root / "real_control.json"

    write_control(visual_control, recording=False, output_dir=None, shutdown=False)
    write_control(real_control, recording=False, output_dir=None, shutdown=False)

    visual_process = subprocess.Popen(
        [
            sys.executable,
            str(VISUAL_SCRIPT),
            "--config",
            "config/apriltag_tracking.yaml",
            "--control-file",
            str(visual_control),
        ],
        cwd=str(VISUAL_WORKSPACE_ROOT),
        text=True,
    )
    real_process = subprocess.Popen(
        [
            sys.executable,
            str(REAL_SCRIPT),
            "--config",
            "config/default.yaml",
            "--control-file",
            str(real_control),
        ],
        cwd=str(REAL_WORKSPACE_ROOT),
        text=True,
    )

    try:
        write_control(visual_control, recording=True, output_dir=visual_output, shutdown=False)
        write_control(real_control, recording=True, output_dir=real_output, shutdown=False)
        wait_for_quit()
    finally:
        write_control(visual_control, recording=False, output_dir=None, shutdown=True)
        write_control(real_control, recording=False, output_dir=None, shutdown=True)
        terminate_process(visual_process)
        terminate_process(real_process)

    solve_calibration(output_root, visual_output, real_output, experiment_id)


if __name__ == "__main__":
    main()
