from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paxini26d_mapping.common import ensure_dir, resolve_from  # noqa: E402
from paxini26d_mapping.dataset import load_mapping_config  # noqa: E402
from paxini26d_mapping.training import train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mapping network from the aligned .pt dataset.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--dataset", default=None, help="Optional explicit .pt dataset path.")
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name used for dataset lookup and model output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_mapping_config(resolve_from(PROJECT_ROOT, args.config))
    dataset_name = f"{args.experiment_name}.pt" if args.experiment_name else config["dataset"]["output_file_name"]
    dataset_path = (
        resolve_from(PROJECT_ROOT, args.dataset)
        if args.dataset
        else resolve_from(PROJECT_ROOT, Path(config["paths"]["dataset_root"]) / dataset_name)
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run prepare_dataset.py first.")

    payload = torch.load(dataset_path, map_location="cpu")
    model_root = ensure_dir(resolve_from(PROJECT_ROOT, config["paths"]["model_root"]))
    summary = train_model(dataset_payload=payload, config=config, output_root=model_root, experiment_name=args.experiment_name)
    evaluation = summary["evaluation"]
    print(f"[INFO] Saved model to {summary['checkpoint_path']}")
    print(f"[INFO] Saved dataset splits to {summary['split_paths']}")
    print(
        "[INFO] Validation metrics: "
        f"MAE={evaluation['validation']['mae']:.6f}, "
        f"RMSE={evaluation['validation']['rmse']:.6f}, "
        f"MAE_gap={evaluation['validation_gap_from_train']['mae_gap']:.6f}, "
        f"RMSE_gap={evaluation['validation_gap_from_train']['rmse_gap']:.6f}"
    )
    print(
        "[INFO] Test metrics: "
        f"MAE={evaluation['test']['mae']:.6f}, "
        f"RMSE={evaluation['test']['rmse']:.6f}, "
        f"MAE_gap={evaluation['test_gap_from_train']['mae_gap']:.6f}, "
        f"RMSE_gap={evaluation['test_gap_from_train']['rmse_gap']:.6f}"
    )


if __name__ == "__main__":
    main()
