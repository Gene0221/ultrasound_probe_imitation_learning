from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DATA_COLLECTION_ROOT = WORKSPACE_ROOT.parent
DEFAULT_VISUAL_LOG = DATA_COLLECTION_ROOT / "visual_pose_tracking" / "output" / "tag_pose_deltas.jsonl"
DEFAULT_REAL_LOG = DATA_COLLECTION_ROOT / "real_pose_tracking" / "output" / "franka_ee_pose_deltas.jsonl"
DEFAULT_OUTPUT_PT = WORKSPACE_ROOT / "output" / "paired_quaternion_dataset.pt"
DEFAULT_SUMMARY_JSON = WORKSPACE_ROOT / "output" / "paired_quaternion_dataset_summary.json"
TIMESTAMP_KEY = "curr_host_timestamp_s"
QUATERNION_KEY = "delta_quaternion_xyzw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired visual/real quaternion dataset.")
    parser.add_argument("--visual-log", type=Path, default=DEFAULT_VISUAL_LOG, help="Path to the visual JSONL log.")
    parser.add_argument("--real-log", type=Path, default=DEFAULT_REAL_LOG, help="Path to the real JSONL log.")
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
        help="Path to the output .pt dataset file.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to the summary JSON file.",
    )
    parser.add_argument(
        "--max-time-diff-s",
        type=float,
        default=0.05,
        help="Maximum allowed nearest-neighbor timestamp gap.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object.")
                records.append(payload)
        return records

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list):
                return [item for item in records if isinstance(item, dict)]
        raise ValueError(f"Unsupported JSON structure in {path}.")

    raise ValueError(f"Unsupported file format for {path}.")


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Encountered zero-norm quaternion.")
    return quaternion / norm


def quaternion_dot_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.dot(lhs, rhs))


def align_quaternion_sign_xyzw(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if quaternion_dot_xyzw(reference, candidate) < 0.0:
        return -candidate
    return candidate


def quaternion_angle_error_rad_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> float:
    aligned_rhs = align_quaternion_sign_xyzw(lhs, rhs)
    dot_value = np.clip(abs(quaternion_dot_xyzw(lhs, aligned_rhs)), 0.0, 1.0)
    return 2.0 * float(np.arccos(dot_value))


def extract_records(path: Path) -> list[dict[str, Any]]:
    raw_records = load_json_records(path)
    extracted: list[dict[str, Any]] = []
    for index, payload in enumerate(raw_records):
        if TIMESTAMP_KEY not in payload or QUATERNION_KEY not in payload:
            raise KeyError(f"Record {index} in {path} is missing required fields.")
        quaternion = np.asarray(payload[QUATERNION_KEY], dtype=np.float64)
        if quaternion.shape != (4,):
            raise ValueError(f"Record {index} in {path} must contain a 4D quaternion.")
        extracted.append(
            {
                "timestamp_s": float(payload[TIMESTAMP_KEY]),
                "quaternion_xyzw": normalize_quaternion_xyzw(quaternion),
            }
        )
    extracted.sort(key=lambda item: item["timestamp_s"])
    return extracted


def find_nearest_real_record(
    timestamp_s: float,
    real_timestamps: list[float],
    real_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    insertion_index = bisect_left(real_timestamps, timestamp_s)
    candidate_indices: list[int] = []
    if insertion_index < len(real_timestamps):
        candidate_indices.append(insertion_index)
    if insertion_index > 0:
        candidate_indices.append(insertion_index - 1)
    if not candidate_indices:
        raise ValueError("Real record list is empty.")

    best_index = min(candidate_indices, key=lambda idx: abs(real_timestamps[idx] - timestamp_s))
    best_record = real_records[best_index]
    return best_record, abs(best_record["timestamp_s"] - timestamp_s)


def summarize_scalar(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": float(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "median": float(np.median(array)) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "p90": float(np.percentile(array, 90.0)) if array.size else 0.0,
        "p95": float(np.percentile(array, 95.0)) if array.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    visual_records = extract_records(args.visual_log)
    real_records = extract_records(args.real_log)
    if not visual_records:
        raise ValueError(f"No visual records found in {args.visual_log}.")
    if not real_records:
        raise ValueError(f"No real records found in {args.real_log}.")

    real_timestamps = [record["timestamp_s"] for record in real_records]

    visual_quaternions: list[np.ndarray] = []
    real_quaternions: list[np.ndarray] = []
    visual_timestamps: list[float] = []
    real_timestamps_matched: list[float] = []
    time_diffs_s: list[float] = []
    angle_errors_rad: list[float] = []
    skipped_count = 0

    for visual_record in visual_records:
        matched_real, time_diff_s = find_nearest_real_record(
            timestamp_s=visual_record["timestamp_s"],
            real_timestamps=real_timestamps,
            real_records=real_records,
        )
        if time_diff_s > args.max_time_diff_s:
            skipped_count += 1
            continue

        visual_quat = visual_record["quaternion_xyzw"]
        real_quat = align_quaternion_sign_xyzw(visual_quat, matched_real["quaternion_xyzw"])
        visual_quaternions.append(visual_quat.astype(np.float32))
        real_quaternions.append(real_quat.astype(np.float32))
        visual_timestamps.append(float(visual_record["timestamp_s"]))
        real_timestamps_matched.append(float(matched_real["timestamp_s"]))
        time_diffs_s.append(float(time_diff_s))
        angle_errors_rad.append(quaternion_angle_error_rad_xyzw(visual_quat, real_quat))

    if not visual_quaternions:
        raise ValueError("No matched samples survived the maximum time-difference filter.")

    dataset = {
        "visual_quat_xyzw": torch.tensor(np.stack(visual_quaternions), dtype=torch.float32),
        "real_quat_xyzw": torch.tensor(np.stack(real_quaternions), dtype=torch.float32),
        "visual_timestamp_s": torch.tensor(visual_timestamps, dtype=torch.float64),
        "real_timestamp_s": torch.tensor(real_timestamps_matched, dtype=torch.float64),
        "time_diff_s": torch.tensor(time_diffs_s, dtype=torch.float64),
        "metadata": {
            "visual_log": str(args.visual_log),
            "real_log": str(args.real_log),
            "timestamp_key": TIMESTAMP_KEY,
            "quaternion_key": QUATERNION_KEY,
            "max_time_diff_s": float(args.max_time_diff_s),
            "matched_samples": len(visual_quaternions),
            "skipped_samples": skipped_count,
        },
    }

    angle_errors_deg = np.rad2deg(np.asarray(angle_errors_rad, dtype=np.float64))
    summary = {
        "visual_log": str(args.visual_log),
        "real_log": str(args.real_log),
        "visual_records": len(visual_records),
        "real_records": len(real_records),
        "matched_samples": len(visual_quaternions),
        "skipped_samples_due_to_time_diff": skipped_count,
        "max_time_diff_s": float(args.max_time_diff_s),
        "time_diff_s_stats": summarize_scalar(time_diffs_s),
        "angle_error_deg_stats": summarize_scalar(angle_errors_deg.tolist()),
    }

    ensure_parent(args.output_pt)
    ensure_parent(args.summary_json)
    torch.save(dataset, args.output_pt)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] Saved paired dataset to: {args.output_pt}")
    print(f"[DONE] Saved summary to: {args.summary_json}")
    print("[INFO] Baseline angle error (deg)")
    print(
        "       "
        f"mean={summary['angle_error_deg_stats']['mean']:.6f}, "
        f"median={summary['angle_error_deg_stats']['median']:.6f}, "
        f"p95={summary['angle_error_deg_stats']['p95']:.6f}, "
        f"max={summary['angle_error_deg_stats']['max']:.6f}"
    )


if __name__ == "__main__":
    main()
