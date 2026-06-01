from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .common import ensure_dir, load_json, save_json


@dataclass
class TimedRecord:
    host_timestamp_s: float
    payload: dict[str, Any]


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            records.append(payload)
    return records


def parse_timed_records(path: Path) -> list[TimedRecord]:
    timed_records: list[TimedRecord] = []
    for record in load_jsonl_records(path):
        if "host_timestamp_s" not in record:
            raise KeyError(f"Missing host_timestamp_s in {path}")
        timed_records.append(TimedRecord(host_timestamp_s=float(record["host_timestamp_s"]), payload=record))
    timed_records.sort(key=lambda item: item.host_timestamp_s)
    return timed_records


def nearest_record(
    timestamp_s: float,
    records: list[TimedRecord],
) -> tuple[Optional[TimedRecord], float]:
    if not records:
        return None, float("inf")
    timestamps = [record.host_timestamp_s for record in records]
    insert_idx = bisect_left(timestamps, timestamp_s)
    candidates: list[TimedRecord] = []
    if insert_idx < len(records):
        candidates.append(records[insert_idx])
    if insert_idx > 0:
        candidates.append(records[insert_idx - 1])
    best = min(candidates, key=lambda item: abs(item.host_timestamp_s - timestamp_s))
    return best, abs(best.host_timestamp_s - timestamp_s)


def build_dataset(
    config: dict[str, Any],
    project_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    paths_cfg = config["paths"]
    dataset_cfg = config["dataset"]
    alignment_cfg = config["alignment"]
    sessions_root = ensure_dir((project_root / paths_cfg["sessions_root"]).resolve())
    max_delta_s = float(alignment_cfg["max_time_delta_s"])

    session_dirs = sorted(path for path in sessions_root.iterdir() if path.is_dir() and path.name.startswith("session_"))
    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    session_ids: list[str] = []
    anchor_timestamps: list[float] = []
    match_deltas: list[dict[str, float]] = []
    feature_dim: Optional[int] = None

    for session_dir in session_dirs:
        imu_path = session_dir / "imu" / dataset_cfg["imu_file_name"]
        left_path = session_dir / "paxini" / dataset_cfg["paxini_left_file_name"]
        right_path = session_dir / "paxini" / dataset_cfg["paxini_right_file_name"]
        force6d_path = session_dir / "force6d" / dataset_cfg["force6d_file_name"]

        if not imu_path.exists() or not left_path.exists() or not right_path.exists() or not force6d_path.exists():
            continue

        imu_records = parse_timed_records(imu_path)
        left_records = parse_timed_records(left_path)
        right_records = parse_timed_records(right_path)
        force_records = parse_timed_records(force6d_path)

        for force_record in force_records:
            anchor_ts = force_record.host_timestamp_s
            imu_record, imu_delta = nearest_record(anchor_ts, imu_records)
            left_record, left_delta = nearest_record(anchor_ts, left_records)
            right_record, right_delta = nearest_record(anchor_ts, right_records)
            if (
                imu_record is None
                or left_record is None
                or right_record is None
                or imu_delta > max_delta_s
                or left_delta > max_delta_s
                or right_delta > max_delta_s
            ):
                continue

            left_values = list(left_record.payload.get("values", []))
            right_values = list(right_record.payload.get("values", []))
            features = [
                float(imu_record.payload["pitch_deg"]),
                float(imu_record.payload["roll_deg"]),
                *[float(value) for value in left_values],
                *[float(value) for value in right_values],
            ]
            if feature_dim is None:
                feature_dim = len(features)
            elif len(features) != feature_dim:
                raise ValueError(
                    f"Inconsistent feature dimension in session {session_dir.name}: "
                    f"expected {feature_dim}, got {len(features)}"
                )

            targets = [float(force_record.payload["Fz"])]
            feature_rows.append(features)
            target_rows.append(targets)
            session_ids.append(session_dir.name)
            anchor_timestamps.append(anchor_ts)
            match_deltas.append(
                {
                    "imu_delta_s": imu_delta,
                    "left_delta_s": left_delta,
                    "right_delta_s": right_delta,
                }
            )

    if not feature_rows:
        raise RuntimeError(
            "No aligned samples were produced. Check session contents, file names, and timestamp coverage."
        )

    left_dim = (feature_dim - 2) // 2 if feature_dim is not None else 0
    right_dim = feature_dim - 2 - left_dim if feature_dim is not None else 0
    feature_names = ["pitch_deg", "roll_deg"]
    feature_names.extend([f"left_{idx}" for idx in range(left_dim)])
    feature_names.extend([f"right_{idx}" for idx in range(right_dim)])

    payload = {
        "features": torch.tensor(feature_rows, dtype=torch.float32),
        "targets": torch.tensor(target_rows, dtype=torch.float32),
        "feature_names": feature_names,
        "target_names": ["Fz"],
        "session_ids": session_ids,
        "anchor_timestamps_s": anchor_timestamps,
        "match_deltas_s": match_deltas,
        "config": config,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    summary = {
        "dataset_path": str(output_path),
        "num_sessions_scanned": len(session_dirs),
        "num_samples": len(feature_rows),
        "feature_dim": int(payload["features"].shape[1]),
        "target_dim": int(payload["targets"].shape[1]),
    }
    save_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def load_mapping_config(config_path: Path) -> dict[str, Any]:
    return load_json(config_path)
