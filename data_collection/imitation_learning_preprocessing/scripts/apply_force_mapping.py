from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_COLLECTION_ROOT = PROJECT_ROOT.parent

# Path to paxini26D_mapping source so we can import its model and utilities.
PAXINI26D_SRC = DATA_COLLECTION_ROOT / "paxini26D_mapping" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PAXINI26D_SRC) not in sys.path:
    sys.path.insert(0, str(PAXINI26D_SRC))

from module.common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    config_get,
    latest_matching_file,
    load_config,
    output_path_for_session,
    resolve_path,
    resolve_session_dirs,
    write_json,
    write_jsonl,
)
from paxini26d_mapping.dataset import nearest_record, parse_timed_records  # noqa: E402
from paxini26d_mapping.training import MLPRegressor  # noqa: E402


def build_feature_values(
    imu_payload: dict[str, Any],
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    feature_names: list[str],
) -> list[float]:
    """Construct the feature vector in the order the model expects.

    Matches the same logic used during training so the column order
    corresponds to ``feature_names`` saved in the checkpoint.
    """
    left_values = [float(value) for value in left_payload.get("values", [])]
    right_values = [float(value) for value in right_payload.get("values", [])]
    feature_map: dict[str, float] = {
        "pitch_deg": float(imu_payload["pitch_deg"]),
        "roll_deg": float(imu_payload["roll_deg"]),
    }
    feature_map.update({f"left_{idx}": value for idx, value in enumerate(left_values)})
    feature_map.update({f"right_{idx}": value for idx, value in enumerate(right_values)})

    missing = [name for name in feature_names if name not in feature_map]
    if missing:
        raise KeyError(f"Missing feature values for model inputs: {missing}")
    return [feature_map[name] for name in feature_names]


def checkpoint_file_from_path(config: dict[str, Any], path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    patterns = config_get(config, "model", "checkpoint_patterns", ["model.pt", "*/model.pt", "run_*/model.pt"])
    return latest_matching_file(path, [str(pattern) for pattern in patterns])


def resolve_checkpoint(config: dict[str, Any], checkpoint_arg: str | None) -> Path:
    """Return the path to a trained model checkpoint.

    If ``checkpoint_arg`` is given it is used directly (relative paths are
    resolved against the data-collection root).  Otherwise the latest
    ``run_*/model.pt`` under ``paxini26D_mapping/model/`` is selected.
    """
    if checkpoint_arg:
        checkpoint_path = resolve_path(checkpoint_arg, base=DATA_COLLECTION_ROOT)
        if checkpoint_path is None:
            raise ValueError("Checkpoint path resolved to None.")
        return checkpoint_file_from_path(config, checkpoint_path)

    configured_checkpoint = config_get(config, "model", "checkpoint")
    if configured_checkpoint:
        checkpoint_path = resolve_path(configured_checkpoint)
        if checkpoint_path is None:
            raise ValueError("Configured checkpoint path resolved to None.")
        return checkpoint_file_from_path(config, checkpoint_path)

    model_root = resolve_path(config_get(config, "paths", "model_root"), base=PROJECT_ROOT)
    if model_root is None:
        model_root = DATA_COLLECTION_ROOT / "paxini26D_mapping" / "model"
    patterns = config_get(config, "model", "checkpoint_patterns", ["*/model.pt", "run_*/model.pt"])
    try:
        return latest_matching_file(model_root.resolve(strict=True), [str(pattern) for pattern in patterns])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No model checkpoint found under {model_root}. "
            "Train the force-mapping model first, or provide an explicit --checkpoint path."
        ) from exc


def load_model(checkpoint_path: Path) -> tuple[MLPRegressor, dict[str, Any]]:
    """Load an MLPRegressor from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_cfg = checkpoint["config"]["training"]
    feature_names = list(checkpoint["feature_names"])
    target_names = list(checkpoint["target_names"])
    model = MLPRegressor(
        input_dim=len(feature_names),
        output_dim=len(target_names),
        hidden_dims=[int(d) for d in training_cfg["hidden_dims"]],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained force-mapping model (IMU + Paxini -> 6D force) and apply it to "
            "a hospital session that contains imu/ and paxini_force/ subdirectories. "
            "IMU records are aligned to Paxini left-sensor timestamps; the predicted 6D "
            "force is written out with the Paxini timestamp."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to preprocessing YAML config file. (default: config/preprocess_dataset.yaml)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Path to the hospital session directory "
            "(e.g. E:/hospital_collection/output/session_0001/)."
        ),
    )
    parser.add_argument(
        "--session-root",
        default=None,
        help=(
            "Path to the hospital output root directory containing session_xxxx/ subdirectories. "
            "All sessions will be processed in batch. "
            "(e.g. E:/hospital_collection/output/). "
            "Mutually exclusive with --session."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a trained model.pt checkpoint.  Relative paths are resolved against "
            "the data-collection root. You may also pass a model run directory. "
            "Defaults to the newest configured model.pt under "
            "paxini26D_mapping/model/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for predicted_force.jsonl and summary.  "
            "Defaults to <session>/predicted_force/."
        ),
    )
    parser.add_argument(
        "--max-time-delta-s",
        type=float,
        default=None,
        help="Maximum allowed timestamp delta (seconds) between Paxini anchor and IMU/right record. (default: 0.05)",
    )
    parser.add_argument(
        "--imu-subdir",
        default=None,
        help="Name of the IMU subdirectory inside the session. (default: imu)",
    )
    parser.add_argument(
        "--paxini-subdir",
        default=None,
        help="Name of the Paxini subdirectory inside the session. (default: paxini_force)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    session_layout = config.get("session_layout", {})
    force_mapping_cfg = config.get("force_mapping", {})
    session_dirs = resolve_session_dirs(
        config=config,
        session_arg=args.session,
        session_root_arg=args.session_root,
    )
    print(f"[INFO] Found {len(session_dirs)} session(s).")

    for session_dir in session_dirs:
        print(f"\n{'='*60}")
        print(f"[INFO] Processing session: {session_dir.name}")
        print(f"{'='*60}")
        _process_session(session_dir, args, config, session_layout, force_mapping_cfg)

    print(f"\n[INFO] All done. Processed {len(session_dirs)} session(s).")


def _process_session(
    session_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    session_layout: dict[str, Any],
    force_mapping_cfg: dict[str, Any],
) -> None:
    output_path = output_path_for_session(
        session_dir=session_dir,
        output_dir_arg=args.output_dir,
        output_subdir=str(force_mapping_cfg.get("output_subdir", "predicted_force")),
        file_name=str(force_mapping_cfg.get("output_file", "predicted_force.jsonl")),
    )
    summary_path = output_path.parent / str(force_mapping_cfg.get("summary_file", "predicted_force.summary.json"))
    max_delta_s = (
        args.max_time_delta_s
        if args.max_time_delta_s is not None
        else float(force_mapping_cfg.get("max_time_delta_s", 0.05))
    )

    # ── Resolve data files ────────────────────────────────────────────
    imu_subdir = args.imu_subdir or session_layout.get("imu_subdir", "imu")
    paxini_subdir = args.paxini_subdir or session_layout.get("paxini_subdir", "paxini_force")
    imu_file = session_layout.get("imu_file", "imu_pitch_roll.jsonl")
    paxini_left_file = session_layout.get("paxini_left_file", "left_sensor.jsonl")
    paxini_right_file = session_layout.get("paxini_right_file", "right_sensor.jsonl")

    imu_path = session_dir / imu_subdir / imu_file
    left_path = session_dir / paxini_subdir / paxini_left_file
    right_path = session_dir / paxini_subdir / paxini_right_file

    for label, path in [("IMU", imu_path), ("Paxini left", left_path), ("Paxini right", right_path)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    # ── Load model ────────────────────────────────────────────────────
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    model, checkpoint = load_model(checkpoint_path)
    feature_mean = checkpoint["feature_mean"].float()
    feature_std = checkpoint["feature_std"].float()
    feature_names = list(checkpoint["feature_names"])
    target_names = list(checkpoint["target_names"])

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Features ({len(feature_names)}): {feature_names}")
    print(f"[INFO] Targets ({len(target_names)}): {target_names}")

    # ── Parse input records ───────────────────────────────────────────
    imu_records = parse_timed_records(imu_path)
    left_records = parse_timed_records(left_path)
    right_records = parse_timed_records(right_path)

    print(f"[INFO] Session: {session_dir.name}")
    print(f"[INFO] Loaded IMU records: {len(imu_records)}")
    print(f"[INFO] Loaded Paxini left records: {len(left_records)}")
    print(f"[INFO] Loaded Paxini right records: {len(right_records)}")

    # ── Align & predict ───────────────────────────────────────────────
    output_records: list[dict[str, Any]] = []
    skipped = 0

    with torch.no_grad():
        for left_record in left_records:
            anchor_ts = left_record.host_timestamp_s

            # Nearest IMU record
            imu_record, imu_delta = nearest_record(anchor_ts, imu_records)
            # Nearest right Paxini record (left & right are captured in pairs,
            # but nearest-record gives a safety check against dropped samples).
            right_record, right_delta = nearest_record(anchor_ts, right_records)

            if (
                imu_record is None
                or right_record is None
                or imu_delta > max_delta_s
                or right_delta > max_delta_s
            ):
                skipped += 1
                continue

            feature_values = build_feature_values(
                imu_record.payload,
                left_record.payload,
                right_record.payload,
                feature_names,
            )
            features = torch.tensor([feature_values], dtype=torch.float32)
            normalized = (features - feature_mean) / feature_std
            predicted_values = model(normalized).squeeze(0).cpu().tolist()
            if not isinstance(predicted_values, list):
                predicted_values = [float(predicted_values)]

            output_records.append(
                {
                    "host_timestamp_s": anchor_ts,
                    "prediction": {
                        name: float(value)
                        for name, value in zip(target_names, predicted_values)
                    },
                    "predicted_values": [float(v) for v in predicted_values],
                    "feature_names": feature_names,
                    "feature_values": feature_values,
                    "match_deltas_s": {
                        "imu_delta_s": float(imu_delta),
                        "right_delta_s": float(right_delta),
                    },
                    "source_timestamps_s": {
                        "imu": imu_record.host_timestamp_s,
                        "left": left_record.host_timestamp_s,
                        "right": right_record.host_timestamp_s,
                    },
                }
            )

    # ── Write output ──────────────────────────────────────────────────
    write_jsonl(output_path, output_records)

    summary = {
        "session_dir": str(session_dir),
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "num_imu_records": len(imu_records),
        "num_left_records": len(left_records),
        "num_right_records": len(right_records),
        "num_predictions": len(output_records),
        "num_skipped": skipped,
        "max_time_delta_s": max_delta_s,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    write_json(summary_path, summary)

    print(f"[INFO] Predictions written: {output_path}")
    print(f"[INFO] Summary: {summary_path}")
    print(f"[INFO] Total predictions: {len(output_records)}, skipped: {skipped}")


if __name__ == "__main__":
    main()
