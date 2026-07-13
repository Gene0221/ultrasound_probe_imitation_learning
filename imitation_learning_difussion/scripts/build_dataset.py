from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diffusion_ultrasound.data_builder import build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diffusion ultrasound action-chunk datasets.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(args.config)
    print(f"[DONE] Diffusion dataset written to {summary['dataset_root']}")


if __name__ == "__main__":
    main()

