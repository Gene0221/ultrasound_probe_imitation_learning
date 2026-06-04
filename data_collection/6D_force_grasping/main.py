from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
HOSPITAL_SRC = WORKSPACE_ROOT.parent / "hospital_data_collection" / "src"
if str(HOSPITAL_SRC) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_SRC))

from hospital_data_collection.launcher import CollectionLauncher


def main() -> None:
    launcher = CollectionLauncher.from_config_path(WORKSPACE_ROOT / "config" / "session_collection.yaml")
    launcher.run_interactive()


if __name__ == "__main__":
    main()
