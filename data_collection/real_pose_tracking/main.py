from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
RUNTIME_SRC = WORKSPACE_ROOT.parent / "collection_runtime" / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from collection_runtime.launcher import CollectionLauncher


def main() -> None:
    launcher = CollectionLauncher.from_config_path(WORKSPACE_ROOT / "config" / "default.yaml")
    launcher.run_interactive()


if __name__ == "__main__":
    main()
