from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hospital_data_collection.launcher import CollectionLauncher


def main() -> None:
    launcher = CollectionLauncher.from_config_path(WORKSPACE_ROOT / "config" / "default.yaml")
    launcher.run_interactive()


if __name__ == "__main__":
    main()
