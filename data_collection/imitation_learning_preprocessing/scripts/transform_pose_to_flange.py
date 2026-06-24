from __future__ import annotations

import argparse
import json
import math
import yaml
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_COLLECTION_ROOT = PROJECT_ROOT.parent
TAG2FLANGE_ROOT = DATA_COLLECTION_ROOT / "tag2flange_calibration"


# ---------------------------------------------------------------------------
# Geometry helpers  (kept local so the script is self-contained)
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [x, y, z, w]."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    rot = transform[:3, :3]
    t = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ t
    return inv


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the nearest valid SO(3) rotation."""
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def sanitize_transform(matrix: np.ndarray) -> np.ndarray:
    """Ensure a 4x4 matrix has a valid SO(3) rotation and clean last row."""
    m = matrix.copy()
    m[:3, :3] = project_rotation_to_so3(m[:3, :3])
    m[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return m


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def latest_calibration_dir(root: Path) -> Path:
    candidates = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("collection_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No collection_* calibration directories found under {root}. "
            "Run tag2flange_calibration first."
        )
    return candidates[-1]


def resolve_calibration(calibration_arg: str | None) -> Path:
    """Return the path to a calibration NPZ bundle.

    If ``calibration_arg`` is given it is used directly (relative paths
    resolved against the data-collection root).  Otherwise the newest
    ``tag2flange_calibration_data.npz`` is selected.
    """
    if calibration_arg:
        cp = Path(calibration_arg)
        if cp.is_absolute():
            return cp.resolve()
        return (DATA_COLLECTION_ROOT / cp).resolve()

    calib_dir = latest_calibration_dir(TAG2FLANGE_ROOT / "output")
    npz_path = calib_dir / "tag2flange_calibration_data.npz"
    if not npz_path.exists():
        json_path = calib_dir / "tag2flange_calibration_report.json"
        if json_path.exists():
            # Fall back to JSON report (contains T_tag_to_flange as nested list)
            return json_path
        raise FileNotFoundError(
            f"Neither tag2flange_calibration_data.npz nor "
            f"tag2flange_calibration_report.json found in {calib_dir}."
        )
    return npz_path


def load_tag_to_flange(calibration_path: Path) -> np.ndarray:
    """Load the 4x4 T_tag_to_flange matrix from NPZ or JSON report."""
    suffix = calibration_path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(calibration_path)
        matrix = payload["T_tag_to_flange"]
    elif suffix == ".json":
        report = json.loads(calibration_path.read_text(encoding="utf-8"))
        matrix = np.asarray(report["estimated_transforms"]["T_tag_to_flange"], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported calibration file format: {suffix}")
    return sanitize_transform(matrix)


# ---------------------------------------------------------------------------
# Visual pose delta loading
# ---------------------------------------------------------------------------

def load_visual_deltas(path: Path) -> list[dict[str, Any]]:
    """Load tag_pose_deltas.jsonl and return a list of records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            if "delta_transform_prev_to_curr" not in payload:
                raise KeyError(f"Missing 'delta_transform_prev_to_curr' at {path}:{line_no}")
            records.append(payload)
    return records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform visual AprilTag pose deltas (tag frame) to flange-frame "
            "pose deltas using a pre-computed tag-to-flange calibration.\n\n"
            "Each output record keeps the original ``curr_host_timestamp_s`` from "
            "the visual delta record."
        )
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Path to the hospital session directory "
            "(e.g. ../hospital_data_collection/output/session_0001/)."
        ),
    )
    parser.add_argument(
        "--session-root",
        default=None,
        help=(
            "Path to the hospital output root directory containing session_xxxx/ subdirectories. "
            "All sessions will be processed in batch. "
            "(e.g. ../hospital_data_collection/output/). "
            "Mutually exclusive with --session."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "preprocess_dataset.yaml"),
        help="Path to preprocessing YAML config file. (default: config/preprocess_dataset.yaml)",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Path to tag2flange calibration NPZ or JSON report. "
            "Relative paths are resolved against the data-collection root. "
            "Defaults to the newest output bundle under tag2flange_calibration/output/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <session>/transformed_pose/.",
    )
    parser.add_argument(
        "--visual-subdir",
        default=None,
        help="Subdirectory name for visual data inside the session. (default: visual_pose)",
    )
    parser.add_argument(
        "--visual-file",
        default=None,
        help="Visual delta file name. (default: tag_pose_deltas.jsonl)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────
    config_path = Path(args.config).resolve()
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        config = {}
        print(f"[WARN] Config file not found: {config_path}, using defaults.")
    session_layout = config.get("session_layout", {})
    pose_transform_cfg = config.get("pose_transform", {})

    # ── Resolve session source (CLI > config > error) ───────────────────
    if args.session and args.session_root:
        parser.error("--session and --session-root are mutually exclusive.")
    if not args.session and not args.session_root:
        config_session_root = config.get("session_root")
        if config_session_root:
            args.session_root = str((PROJECT_ROOT / config_session_root).resolve())
        else:
            parser.error(
                "Either --session or --session-root must be provided, "
                "or set session_root in config/preprocess_dataset.yaml."
            )

    # ── Resolve session(s) ─────────────────────────────────────────────
    session_dirs: list[Path] = []
    if args.session_root:
        root = Path(args.session_root).resolve(strict=True)
        session_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("session_")],
        )
        if not session_dirs:
            raise FileNotFoundError(f"No session_xxxx directories found under {root}")
        print(f"[INFO] Found {len(session_dirs)} sessions under {root}")
    else:
        session_dirs = [Path(args.session).resolve(strict=True)]

    # ── Load calibration ──────────────────────────────────────────────
    calibration_path = resolve_calibration(args.calibration)
    T_tag_to_flange = load_tag_to_flange(calibration_path)
    T_flange_to_tag = invert_transform(T_tag_to_flange)

    print(f"[INFO] Calibration: {calibration_path}")
    print(f"[INFO] T_tag_to_flange rotation angle (deg): "
          f"{math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(T_tag_to_flange[:3, :3]) - 1.0) / 2.0)))):.4f}")
    print(f"[INFO] T_tag_to_flange translation: {T_tag_to_flange[:3, 3].tolist()}")

    # ── Process each session ───────────────────────────────────────────
    for session_idx, session_dir in enumerate(session_dirs, start=1):
        print(f"\n{'='*60}")
        print(f"[INFO] Processing session ({session_idx}/{len(session_dirs)}): {session_dir.name}")
        print(f"{'='*60}")

        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else session_dir / pose_transform_cfg.get("output_subdir", "transformed_pose")
        )

        # ── Load visual deltas ────────────────────────────────────────────
        visual_subdir = args.visual_subdir or session_layout.get("visual_subdir", "visual_pose")
        visual_file = args.visual_file or session_layout.get("visual_file", "tag_pose_deltas.jsonl")
        visual_path = session_dir / visual_subdir / visual_file
        if not visual_path.exists():
            print(f"[WARN] Visual pose file not found: {visual_path}, skipping session.")
            continue
        visual_records = load_visual_deltas(visual_path)
        print(f"[INFO] Session: {session_dir.name}")
        print(f"[INFO] Visual records loaded: {len(visual_records)}")
        print(f"[INFO] Calibration: {calibration_path.name}")

        # ── Transform ─────────────────────────────────────────────────────
        output_records: list[dict[str, Any]] = []
        skipped = 0

        for idx, record in enumerate(visual_records):
            tag_delta = np.asarray(record["delta_transform_prev_to_curr"], dtype=np.float64)
            if tag_delta.shape != (4, 4):
                print(f"[WARN] Record {idx}: delta_transform is not 4x4 (shape={tag_delta.shape}), skipping.")
                skipped += 1
                continue
            tag_delta = sanitize_transform(tag_delta)

            # flange_delta = T_flange_to_tag @ tag_delta @ T_tag_to_flange
            flange_delta = T_flange_to_tag @ tag_delta @ T_tag_to_flange
            flange_delta = sanitize_transform(flange_delta)

            translation = flange_delta[:3, 3].tolist()
            quaternion = rotation_matrix_to_quaternion(flange_delta[:3, :3]).tolist()

            # Preserve all original fields, override the transformed ones.
            output_record = dict(record)
            output_record["delta_transform_prev_to_curr"] = flange_delta.tolist()
            output_record["delta_translation_xyz"] = translation
            output_record["delta_quaternion_xyzw"] = quaternion
            output_record["host_timestamp_s"] = float(record["curr_host_timestamp_s"])
            output_record["source"] = "transformed_from_tag_to_flange"
            output_record["calibration"] = str(calibration_path)
            output_records.append(output_record)

        # ── Write output ──────────────────────────────────────────────────
        jsonl_path = output_dir / "flange_pose_deltas.jsonl"
        summary_path = output_dir / "flange_pose_deltas.summary.json"

        write_jsonl(jsonl_path, output_records)

        summary = {
            "session_dir": str(session_dir),
            "calibration_path": str(calibration_path),
            "output_path": str(jsonl_path),
            "num_visual_records": len(visual_records),
            "num_transformed": len(output_records),
            "num_skipped": skipped,
            "session_index": session_idx,
            "total_sessions": len(session_dirs),
            "T_tag_to_flange": T_tag_to_flange.tolist(),
        }
        write_summary(summary_path, summary)

        print(f"[INFO] Transformed records written: {jsonl_path}")
        print(f"[INFO] Total transformed: {len(output_records)}, skipped: {skipped}")

    print(f"\n[INFO] All done. Processed {len(session_dirs)} session(s).")


if __name__ == "__main__":
    main()
