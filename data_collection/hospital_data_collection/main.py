from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT.parent / "collection_runtime" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collection_runtime.launcher import CollectionLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital integrated data collection launcher.")
    parser.add_argument("--command-file", default=None, help="Optional JSON command file for non-interactive control.")
    parser.add_argument("--status-file", default=None, help="Optional JSON status file written by the launcher.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launcher = CollectionLauncher.from_config_path(WORKSPACE_ROOT / "config" / "default.yaml")
    if args.command_file:
        launcher.run_command_file(command_file=args.command_file, status_file=args.status_file)
    else:
        launcher.run_interactive()


if __name__ == "__main__":
    main()
