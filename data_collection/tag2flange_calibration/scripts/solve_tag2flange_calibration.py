from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DATA_COLLECTION_ROOT = WORKSPACE_ROOT.parent
DEFAULT_VISUAL_LOG = DATA_COLLECTION_ROOT / "visual_pose_tracking" / "output" / "tag_pose_deltas.jsonl"
DEFAULT_REAL_LOG = DATA_COLLECTION_ROOT / "real_pose_tracking" / "output" / "franka_ee_pose_deltas.jsonl"
DEFAULT_OUTPUT_JSON = WORKSPACE_ROOT / "output" / "tag2flange_calibration_report.json"
DEFAULT_OUTPUT_NPZ = WORKSPACE_ROOT / "output" / "tag2flange_calibration_data.npz"
TIMESTAMP_KEY = "curr_host_timestamp_s"
TRANSFORM_KEY = "delta_transform_prev_to_curr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve tag-to-flange hand-eye calibration from paired relative motions.")
    parser.add_argument("--visual-log", type=Path, default=DEFAULT_VISUAL_LOG, help="Path to the visual JSONL log.")
    parser.add_argument("--real-log", type=Path, default=DEFAULT_REAL_LOG, help="Path to the real JSONL log.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Path to the JSON report.")
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ, help="Path to the NPZ bundle.")
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
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return [item for item in payload["records"] if isinstance(item, dict)]
        raise ValueError(f"Unsupported JSON structure in {path}.")

    raise ValueError(f"Unsupported file format for {path}.")


def as_transform_matrix(payload: Any, path: Path, index: int) -> np.ndarray:
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Record {index} in {path} must contain a 4x4 transform.")
    return matrix


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace_value = np.trace(rotation)
    cos_theta = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def extract_records(path: Path) -> list[dict[str, Any]]:
    raw_records = load_json_records(path)
    extracted: list[dict[str, Any]] = []
    for index, payload in enumerate(raw_records):
        if TIMESTAMP_KEY not in payload or TRANSFORM_KEY not in payload:
            raise KeyError(f"Record {index} in {path} is missing required fields.")
        transform = as_transform_matrix(payload[TRANSFORM_KEY], path, index)
        transform = transform.copy()
        transform[:3, :3] = project_rotation_to_so3(transform[:3, :3])
        transform[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        extracted.append(
            {
                "timestamp_s": float(payload[TIMESTAMP_KEY]),
                "transform": transform,
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


def solve_hand_eye_rotation(rotations_a: np.ndarray, rotations_b: np.ndarray) -> np.ndarray:
    blocks = [np.kron(np.eye(3), rotation_a) - np.kron(rotation_b.T, np.eye(3)) for rotation_a, rotation_b in zip(rotations_a, rotations_b)]
    system = np.concatenate(blocks, axis=0)
    _, _, vt = np.linalg.svd(system)
    rotation = vt[-1].reshape(3, 3)
    return project_rotation_to_so3(rotation)


def solve_hand_eye_translation(
    rotations_a: np.ndarray,
    translations_a: np.ndarray,
    rotations_b: np.ndarray,
    translations_b: np.ndarray,
    rotation_x: np.ndarray,
) -> np.ndarray:
    lhs_blocks = [rotation_a - np.eye(3) for rotation_a in rotations_a]
    rhs_blocks = [rotation_x @ translation_b - translation_a for translation_a, translation_b in zip(translations_a, translations_b)]
    lhs = np.concatenate(lhs_blocks, axis=0)
    rhs = np.concatenate(rhs_blocks, axis=0)
    solution, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return solution


def solve_hand_eye(transform_pairs_a: np.ndarray, transform_pairs_b: np.ndarray) -> np.ndarray:
    rotations_a = transform_pairs_a[:, :3, :3]
    translations_a = transform_pairs_a[:, :3, 3]
    rotations_b = transform_pairs_b[:, :3, :3]
    translations_b = transform_pairs_b[:, :3, 3]

    rotation_x = solve_hand_eye_rotation(rotations_a, rotations_b)
    translation_x = solve_hand_eye_translation(
        rotations_a=rotations_a,
        translations_a=translations_a,
        rotations_b=rotations_b,
        translations_b=translations_b,
        rotation_x=rotation_x,
    )

    transform_x = np.eye(4, dtype=np.float64)
    transform_x[:3, :3] = rotation_x
    transform_x[:3, 3] = translation_x
    return transform_x


def summarize_scalar(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "count": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def transform_to_list(transform: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in transform.tolist()]


def main() -> None:
    args = parse_args()
    visual_records = extract_records(args.visual_log)
    real_records = extract_records(args.real_log)
    if not visual_records:
        raise ValueError(f"No visual records found in {args.visual_log}.")
    if not real_records:
        raise ValueError(f"No real records found in {args.real_log}.")

    real_timestamps = [record["timestamp_s"] for record in real_records]
    transform_pairs_a: list[np.ndarray] = []
    transform_pairs_b: list[np.ndarray] = []
    visual_timestamps: list[float] = []
    real_timestamps_matched: list[float] = []
    time_diffs_s: list[float] = []
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

        transform_pairs_a.append(visual_record["transform"])
        transform_pairs_b.append(matched_real["transform"])
        visual_timestamps.append(float(visual_record["timestamp_s"]))
        real_timestamps_matched.append(float(matched_real["timestamp_s"]))
        time_diffs_s.append(float(time_diff_s))

    if len(transform_pairs_a) < 3:
        raise ValueError("At least 3 matched motion pairs are required for calibration.")

    transforms_a = np.stack(transform_pairs_a)
    transforms_b = np.stack(transform_pairs_b)
    tag_to_flange = solve_hand_eye(transforms_a, transforms_b)
    flange_to_tag = invert_transform(tag_to_flange)

    rotation_residuals_deg: list[float] = []
    translation_residuals_m: list[float] = []
    frobenius_residuals: list[float] = []
    for transform_a, transform_b in zip(transforms_a, transforms_b):
        lhs = transform_a @ tag_to_flange
        rhs = tag_to_flange @ transform_b
        delta = invert_transform(lhs) @ rhs
        delta[:3, :3] = project_rotation_to_so3(delta[:3, :3])
        rotation_residuals_deg.append(rotation_angle_deg(delta[:3, :3]))
        translation_residuals_m.append(float(np.linalg.norm(delta[:3, 3])))
        frobenius_residuals.append(float(np.linalg.norm(lhs - rhs, ord="fro")))

    time_diffs_array = np.asarray(time_diffs_s, dtype=np.float64)
    rotation_residuals_array = np.asarray(rotation_residuals_deg, dtype=np.float64)
    translation_residuals_array = np.asarray(translation_residuals_m, dtype=np.float64)
    frobenius_residuals_array = np.asarray(frobenius_residuals, dtype=np.float64)

    report = {
        "visual_log": str(args.visual_log),
        "real_log": str(args.real_log),
        "timestamp_key": TIMESTAMP_KEY,
        "transform_key": TRANSFORM_KEY,
        "visual_records": len(visual_records),
        "real_records": len(real_records),
        "matched_samples": len(transform_pairs_a),
        "skipped_samples_due_to_time_diff": skipped_count,
        "max_time_diff_s": float(args.max_time_diff_s),
        "time_diff_s_stats": summarize_scalar(time_diffs_array),
        "estimated_transforms": {
            "T_tag_to_flange": transform_to_list(tag_to_flange),
            "T_flange_to_tag": transform_to_list(flange_to_tag),
        },
        "residuals": {
            "rotation_deg_stats": summarize_scalar(rotation_residuals_array),
            "translation_m_stats": summarize_scalar(translation_residuals_array),
            "frobenius_stats": summarize_scalar(frobenius_residuals_array),
        },
    }

    ensure_parent(args.output_json)
    ensure_parent(args.output_npz)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(
        args.output_npz,
        visual_transforms=transforms_a,
        real_transforms=transforms_b,
        visual_timestamps=np.asarray(visual_timestamps, dtype=np.float64),
        real_timestamps=np.asarray(real_timestamps_matched, dtype=np.float64),
        time_diff_s=time_diffs_array,
        T_tag_to_flange=tag_to_flange,
        T_flange_to_tag=flange_to_tag,
    )

    print(f"[DONE] Saved calibration report to: {args.output_json}")
    print(f"[DONE] Saved calibration bundle to: {args.output_npz}")
    print("[INFO] Residual summary")
    print(
        "       "
        f"rotation_mean_deg={report['residuals']['rotation_deg_stats']['mean']:.6f}, "
        f"translation_mean_m={report['residuals']['translation_m_stats']['mean']:.6f}, "
        f"rotation_p95_deg={report['residuals']['rotation_deg_stats']['p95']:.6f}, "
        f"translation_p95_m={report['residuals']['translation_m_stats']['p95']:.6f}"
    )


if __name__ == "__main__":
    main()
