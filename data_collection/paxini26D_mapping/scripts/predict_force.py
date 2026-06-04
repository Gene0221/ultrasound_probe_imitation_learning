from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paxini26d_mapping.common import ensure_dir, load_config_file, resolve_from, save_json  # noqa: E402
from paxini26d_mapping.dataset import nearest_record, parse_timed_records  # noqa: E402
from paxini26d_mapping.training import MLPRegressor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained Paxini+IMU mapping model on collected inference inputs."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument(
        "--session",
        default=None,
        help="Inference session directory or name. Defaults to the newest inference session.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit model.pt path. Defaults to the newest model/run_*/model.pt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <session>/prediction/predicted_force.jsonl.",
    )
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--max-time-delta-s",
        type=float,
        default=None,
        help="Maximum allowed timestamp mismatch for IMU/right Paxini records.",
    )
    return parser.parse_args()


def latest_child_dir(root: Path, prefix: str) -> Path:
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No directory starting with '{prefix}' found in {root}")
    return candidates[-1]


def latest_checkpoint(model_root: Path) -> Path:
    candidates = sorted(model_root.glob("run_*/model.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model.pt found under {model_root}")
    return candidates[-1]


def resolve_session(config: dict[str, Any], session_arg: Optional[str]) -> Path:
    inference_cfg = config.get("inference", {})
    sessions_root = resolve_from(PROJECT_ROOT, inference_cfg.get("sessions_root", "inference_sessions"))
    if session_arg:
        session_path = Path(session_arg)
        if session_path.is_absolute() or session_path.exists():
            return session_path.resolve()
        return (sessions_root / session_arg).resolve()
    return latest_child_dir(sessions_root, str(inference_cfg.get("session_prefix", "inference_")))


def resolve_checkpoint(config: dict[str, Any], checkpoint_arg: Optional[str]) -> Path:
    if checkpoint_arg:
        return resolve_from(PROJECT_ROOT, checkpoint_arg)
    return latest_checkpoint(resolve_from(PROJECT_ROOT, config["paths"]["model_root"]))


def build_feature_values(
    imu_payload: dict[str, Any],
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    feature_names: list[str],
) -> list[float]:
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


def load_model(checkpoint_path: Path) -> tuple[MLPRegressor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    training_cfg = config["training"]
    feature_names = list(checkpoint["feature_names"])
    target_names = list(checkpoint["target_names"])
    model = MLPRegressor(
        input_dim=len(feature_names),
        output_dim=len(target_names),
        hidden_dims=[int(dim) for dim in training_cfg["hidden_dims"]],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]], target_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "host_timestamp_s",
        "imu_delta_s",
        "right_delta_s",
        *[f"predicted_{name}" for name in target_names],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "host_timestamp_s": record["host_timestamp_s"],
                "imu_delta_s": record["match_deltas_s"]["imu_delta_s"],
                "right_delta_s": record["match_deltas_s"]["right_delta_s"],
            }
            row.update(
                {
                    f"predicted_{name}": record["prediction"][name]
                    for name in target_names
                }
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config_file(resolve_from(PROJECT_ROOT, args.config))
    dataset_cfg = config["dataset"]
    max_delta_s = (
        float(args.max_time_delta_s)
        if args.max_time_delta_s is not None
        else float(config["alignment"]["max_time_delta_s"])
    )

    session_dir = resolve_session(config, args.session)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    output_path = (
        resolve_from(PROJECT_ROOT, args.output)
        if args.output
        else ensure_dir(session_dir / "prediction") / "predicted_force.jsonl"
    )
    csv_output_path = resolve_from(PROJECT_ROOT, args.csv_output) if args.csv_output else None

    imu_path = session_dir / "imu" / dataset_cfg["imu_file_name"]
    left_path = session_dir / "paxini" / dataset_cfg["paxini_left_file_name"]
    right_path = session_dir / "paxini" / dataset_cfg["paxini_right_file_name"]
    for path in (imu_path, left_path, right_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing inference input file: {path}")

    model, checkpoint = load_model(checkpoint_path)
    feature_mean = checkpoint["feature_mean"].float()
    feature_std = checkpoint["feature_std"].float()
    feature_names = list(checkpoint["feature_names"])
    target_names = list(checkpoint["target_names"])

    imu_records = parse_timed_records(imu_path)
    left_records = parse_timed_records(left_path)
    right_records = parse_timed_records(right_path)

    output_records: list[dict[str, Any]] = []
    skipped = 0
    with torch.no_grad():
        for left_record in left_records:
            anchor_ts = left_record.host_timestamp_s
            imu_record, imu_delta = nearest_record(anchor_ts, imu_records)
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
            normalized_features = (features - feature_mean) / feature_std
            predicted_values = model(normalized_features).squeeze(0).cpu().tolist()
            if not isinstance(predicted_values, list):
                predicted_values = [float(predicted_values)]

            prediction = {
                name: float(value)
                for name, value in zip(target_names, predicted_values)
            }
            output_records.append(
                {
                    "host_timestamp_s": anchor_ts,
                    "prediction": prediction,
                    "predicted_values": [float(value) for value in predicted_values],
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

    write_jsonl(output_path, output_records)
    if csv_output_path is not None:
        write_csv(csv_output_path, output_records, target_names)

    summary = {
        "session_dir": str(session_dir),
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "csv_output_path": str(csv_output_path) if csv_output_path else None,
        "num_predictions": len(output_records),
        "num_skipped": skipped,
        "max_time_delta_s": max_delta_s,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    save_json(output_path.with_suffix(".summary.json"), summary)
    print(f"[INFO] Wrote predictions: {output_path}")
    print(f"[INFO] Predictions: {len(output_records)}, skipped: {skipped}")


if __name__ == "__main__":
    main()
