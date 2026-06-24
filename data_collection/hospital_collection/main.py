from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital integrated data collection console.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run as headless backend (interactive console, no GUI).",
    )
    parser.add_argument(
        "--command-file",
        default=None,
        help="JSON command file for non-interactive control (only in headless mode).",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="JSON status file (only in headless mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.headless:
        _run_headless(args.command_file, args.status_file)
    else:
        _run_gui()


def _run_gui() -> None:
    from hospital_collection.app import run
    run()


def _run_headless(command_file: str | None, status_file: str | None) -> None:
    from hospital_collection.backend_controller import BackendController

    config_path = WORKSPACE_ROOT / "config" / "default.yaml"
    controller = BackendController(config_path)

    print("[INFO] Initialising hospital backend in headless mode...")
    success, message = controller.initialize()
    if not success:
        print(f"[ERROR] Initialisation failed: {message}")
        sys.exit(1)
    print("[INFO] Initialisation complete.")

    if command_file:
        import json
        import time

        from collection_runtime.launcher import CollectionLauncher  # noqa: F811

        status_path = Path(status_file).resolve() if status_file else None

        print(f"[INFO] Command-file control enabled: {command_file}")

        last_sequence = -1
        command_path = Path(command_file).resolve()
        while controller._launcher.state_machine.state.value != "stopped":  # type: ignore[union-attr]
            if command_path.exists():
                try:
                    payload = json.loads(command_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    sequence = int(payload.get("sequence", -1))
                    command = str(payload.get("command", "")).strip().lower()
                    if sequence > last_sequence and command:
                        last_sequence = sequence
                        if command == "start_or_resume":
                            controller.start_or_resume()
                        elif command == "pause":
                            controller.pause()
                        elif command == "quit":
                            break
            time.sleep(0.1)

        controller.shutdown()
    else:
        try:
            while True:
                user_input = input("\nPress Enter to start/pause/resume, q to quit: ")
                if user_input.lower() == "q":
                    break
                controller.start_or_resume() if controller._launcher.state_machine.state.value in {"idle", "paused"} else controller.pause()  # type: ignore[union-attr]
        except KeyboardInterrupt:
            pass
        finally:
            controller.shutdown()


if __name__ == "__main__":
    main()
