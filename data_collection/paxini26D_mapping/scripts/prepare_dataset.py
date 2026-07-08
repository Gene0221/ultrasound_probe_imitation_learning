from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paxini26d_mapping.common import ensure_dir, resolve_from  # noqa: E402
from paxini26d_mapping.dataset import build_dataset, load_mapping_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align all sessions and export a training dataset.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name used in the dataset file name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_from(PROJECT_ROOT, args.config)
    config = load_mapping_config(config_path)
    dataset_root = ensure_dir(resolve_from(PROJECT_ROOT, config["paths"]["dataset_root"]))
    output_name = (
        f"{args.experiment_name}.pt"
        if args.experiment_name
        else config["dataset"]["output_file_name"]
    )
    output_path = dataset_root / output_name
    summary = build_dataset(config=config, project_root=PROJECT_ROOT, output_path=output_path)
    print(f"[INFO] Saved dataset to {output_path}")
    print(f"[INFO] Samples: {summary['num_samples']}, feature_dim: {summary['feature_dim']}")


if __name__ == "__main__":
    main()
