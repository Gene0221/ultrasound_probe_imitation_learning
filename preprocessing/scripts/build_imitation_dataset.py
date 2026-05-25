from __future__ import annotations

import argparse
import csv
import json
import shutil
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]


@dataclass
class PoseRecord:
    timestamp_s: float
    pose_delta_7d: list[float]
    raw_payload: dict[str, Any]


@dataclass
class ForceRecord:
    timestamp_s: float
    force: list[float]
    raw_payload: dict[str, Any]


@dataclass
class UltrasoundRecord:
    timestamp_s: float
    image_path: Path
    raw_payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an imitation-learning dataset by aligning pose, force, and ultrasound data."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing YAML config.")
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return payload


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_records_payload(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {path} line {line_no}.")
                records.append(payload)
        return records

    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            if not all(isinstance(item, dict) for item in payload):
                raise ValueError(f"Expected list of objects in {path}.")
            return payload
        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                if not all(isinstance(item, dict) for item in payload["records"]):
                    raise ValueError(f"Expected list of objects in {path}['records'].")
                return payload["records"]
            raise ValueError(f"JSON file {path} must be a list or contain a top-level 'records' list.")
        raise ValueError(f"Unsupported JSON structure in {path}.")

    raise ValueError(f"Unsupported file type for record loading: {path}")


def parse_pose_records(path: Path) -> list[PoseRecord]:
    records = load_records_payload(path)
    parsed: list[PoseRecord] = []
    for idx, payload in enumerate(records):
        timestamp = float(payload["curr_host_timestamp_s"])
        translation = [float(value) for value in payload["delta_translation_xyz"]]
        quaternion = [float(value) for value in payload["delta_quaternion_xyzw"]]
        if len(translation) != 3:
            raise ValueError(f"Pose record {idx} in {path} has translation dim {len(translation)}, expected 3.")
        if len(quaternion) != 4:
            raise ValueError(f"Pose record {idx} in {path} has quaternion dim {len(quaternion)}, expected 4.")
        parsed.append(
            PoseRecord(
                timestamp_s=timestamp,
                pose_delta_7d=translation + quaternion,
                raw_payload=payload,
            )
        )
    parsed.sort(key=lambda record: record.timestamp_s)
    return parsed


def parse_force_vector(payload: dict[str, Any], force_dim: int, source_name: str) -> list[float]:
    if "force" in payload:
        force = payload["force"]
        if isinstance(force, str):
            force = json.loads(force)
        values = [float(value) for value in force]
    else:
        candidate_columns = ["fx", "fy", "fz", "mx", "my", "mz"]
        present = [column for column in candidate_columns if column in payload and payload[column] not in ("", None)]
        if not present:
            raise ValueError(f"{source_name} is missing a 'force' field and scalar force columns.")
        values = [float(payload[column]) for column in present]
    if len(values) != force_dim:
        raise ValueError(f"{source_name} force dim {len(values)} does not match configured force_dim={force_dim}.")
    return values


def parse_force_records(path: Path, force_dim: int) -> list[ForceRecord]:
    suffix = path.suffix.lower()
    parsed: list[ForceRecord] = []

    if suffix in {".json", ".jsonl"}:
        records = load_records_payload(path)
        for idx, payload in enumerate(records):
            parsed.append(
                ForceRecord(
                    timestamp_s=float(payload["host_timestamp_s"]),
                    force=parse_force_vector(payload, force_dim=force_dim, source_name=f"{path} record {idx}"),
                    raw_payload=payload,
                )
            )
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                parsed.append(
                    ForceRecord(
                        timestamp_s=float(row["host_timestamp_s"]),
                        force=parse_force_vector(row, force_dim=force_dim, source_name=f"{path} row {idx + 2}"),
                        raw_payload=dict(row),
                    )
                )
    else:
        raise ValueError(f"Unsupported force file type: {path}")

    parsed.sort(key=lambda record: record.timestamp_s)
    return parsed


def parse_ultrasound_records(images_dir: Path, timestamps_path: Path) -> list[UltrasoundRecord]:
    suffix = timestamps_path.suffix.lower()
    parsed: list[UltrasoundRecord] = []

    def resolve_image_path(image_value: str) -> Path:
        image_path = Path(image_value)
        if image_path.is_absolute():
            return image_path.resolve()
        return (images_dir / image_path).resolve()

    if suffix in {".json", ".jsonl"}:
        records = load_records_payload(timestamps_path)
        for idx, payload in enumerate(records):
            image_path = resolve_image_path(str(payload["image"]))
            parsed.append(
                UltrasoundRecord(
                    timestamp_s=float(payload["host_timestamp_s"]),
                    image_path=image_path,
                    raw_payload=payload,
                )
            )
    elif suffix == ".csv":
        with timestamps_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                image_path = resolve_image_path(str(row["image"]))
                parsed.append(
                    UltrasoundRecord(
                        timestamp_s=float(row["host_timestamp_s"]),
                        image_path=image_path,
                        raw_payload=dict(row),
                    )
                )
    else:
        raise ValueError(f"Unsupported ultrasound timestamp file type: {timestamps_path}")

    parsed.sort(key=lambda record: record.timestamp_s)
    return parsed


def find_nearest_index(sorted_timestamps: list[float], target: float) -> int | None:
    if not sorted_timestamps:
        return None
    insert_idx = bisect_left(sorted_timestamps, target)
    if insert_idx == 0:
        return 0
    if insert_idx >= len(sorted_timestamps):
        return len(sorted_timestamps) - 1

    prev_idx = insert_idx - 1
    next_idx = insert_idx
    prev_dt = abs(sorted_timestamps[prev_idx] - target)
    next_dt = abs(sorted_timestamps[next_idx] - target)
    return prev_idx if prev_dt <= next_dt else next_idx


def copy_image(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Ultrasound image not found: {src}")
    shutil.copy2(src, dst)


def build_trajectory(
    trajectory_cfg: dict[str, Any],
    output_root: Path,
    manifests_root: Path,
    force_dim: int,
    max_pose_force_dt_s: float,
    max_pose_image_dt_s: float,
) -> dict[str, Any]:
    trajectory_id = str(trajectory_cfg["trajectory_id"])
    trajectory_root = ensure_dir(output_root / trajectory_id)
    images_root = ensure_dir(trajectory_root / "images")

    pose_path = resolve_path(trajectory_cfg["pose"]["path"])
    force_path = resolve_path(trajectory_cfg["force"]["path"])
    ultrasound_images_dir = resolve_path(trajectory_cfg["ultrasound"]["images_dir"])
    ultrasound_timestamps_path = resolve_path(trajectory_cfg["ultrasound"]["timestamps_path"])

    pose_records = parse_pose_records(pose_path)
    force_records = parse_force_records(force_path, force_dim=force_dim)
    ultrasound_records = parse_ultrasound_records(
        images_dir=ultrasound_images_dir,
        timestamps_path=ultrasound_timestamps_path,
    )

    force_timestamps = [record.timestamp_s for record in force_records]
    ultrasound_timestamps = [record.timestamp_s for record in ultrasound_records]

    frames: list[dict[str, Any]] = []
    skipped_missing_match = 0
    skipped_threshold = 0

    for pose_record in pose_records:
        force_idx = find_nearest_index(force_timestamps, pose_record.timestamp_s)
        image_idx = find_nearest_index(ultrasound_timestamps, pose_record.timestamp_s)
        if force_idx is None or image_idx is None:
            skipped_missing_match += 1
            continue

        force_record = force_records[force_idx]
        ultrasound_record = ultrasound_records[image_idx]
        pose_force_dt_s = abs(force_record.timestamp_s - pose_record.timestamp_s)
        pose_image_dt_s = abs(ultrasound_record.timestamp_s - pose_record.timestamp_s)
        if pose_force_dt_s > max_pose_force_dt_s or pose_image_dt_s > max_pose_image_dt_s:
            skipped_threshold += 1
            continue

        frame_idx = len(frames)
        output_image_name = f"{frame_idx:06d}{ultrasound_record.image_path.suffix.lower() or '.png'}"
        copy_image(ultrasound_record.image_path, images_root / output_image_name)

        frames.append(
            {
                "image": f"images/{output_image_name}",
                "pose_delta_7d": pose_record.pose_delta_7d,
                "force": force_record.force,
                "pose_timestamp_s": pose_record.timestamp_s,
                "force_timestamp_s": force_record.timestamp_s,
                "image_timestamp_s": ultrasound_record.timestamp_s,
                "pose_force_dt_s": pose_force_dt_s,
                "pose_image_dt_s": pose_image_dt_s,
            }
        )

    metadata = {
        "trajectory_id": trajectory_id,
        "frames": frames,
        "source_files": {
            "pose_path": str(pose_path),
            "force_path": str(force_path),
            "ultrasound_images_dir": str(ultrasound_images_dir),
            "ultrasound_timestamps_path": str(ultrasound_timestamps_path),
        },
        "alignment_summary": {
            "pose_records_total": len(pose_records),
            "force_records_total": len(force_records),
            "ultrasound_records_total": len(ultrasound_records),
            "frames_exported": len(frames),
            "skipped_missing_match": skipped_missing_match,
            "skipped_threshold": skipped_threshold,
            "max_pose_force_dt_s": max_pose_force_dt_s,
            "max_pose_image_dt_s": max_pose_image_dt_s,
        },
    }
    write_json(trajectory_root / "metadata.json", metadata)

    return {
        "trajectory_id": trajectory_id,
        "split": str(trajectory_cfg.get("split", "train")),
        "root_dir": str(trajectory_root.relative_to(manifests_root.parent)).replace("\\", "/"),
        "metadata_path": str((trajectory_root / "metadata.json").relative_to(manifests_root.parent)).replace("\\", "/"),
        "num_frames": len(frames),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))

    dataset_cfg = config["dataset"]
    alignment_cfg = config["alignment"]
    trajectories_cfg = config["trajectories"]
    if not isinstance(trajectories_cfg, list) or not trajectories_cfg:
        raise ValueError("Config must define a non-empty 'trajectories' list.")

    dataset_root = ensure_dir(resolve_path(dataset_cfg["output_dir"]))
    manifests_root = ensure_dir(dataset_root / str(dataset_cfg.get("manifests_dir", "manifests")))
    trajectories_root = ensure_dir(dataset_root / str(dataset_cfg.get("trajectories_dir", "trajectories")))

    use_force = bool(dataset_cfg.get("use_force", True))
    force_dim = int(dataset_cfg["force_dim"])
    max_pose_force_dt_s = float(alignment_cfg["max_pose_force_dt_s"])
    max_pose_image_dt_s = float(alignment_cfg["max_pose_image_dt_s"])

    train_trajectories: list[dict[str, Any]] = []
    val_trajectories: list[dict[str, Any]] = []

    for trajectory_cfg in trajectories_cfg:
        built = build_trajectory(
            trajectory_cfg=trajectory_cfg,
            output_root=trajectories_root,
            manifests_root=manifests_root,
            force_dim=force_dim,
            max_pose_force_dt_s=max_pose_force_dt_s,
            max_pose_image_dt_s=max_pose_image_dt_s,
        )
        manifest_entry = {
            "trajectory_id": built["trajectory_id"],
            "root_dir": built["root_dir"],
            "metadata_path": built["metadata_path"],
        }
        split = built["split"].lower()
        if split == "train":
            train_trajectories.append(manifest_entry)
        elif split == "val":
            val_trajectories.append(manifest_entry)
        else:
            raise ValueError(f"Unsupported split '{built['split']}' for trajectory {built['trajectory_id']}.")

        print(
            f"[{split}] {built['trajectory_id']}: exported {built['num_frames']} aligned frames to "
            f"{trajectories_root / built['trajectory_id']}"
        )

    manifest_payload = {
        "use_force": use_force,
        "force_dim": force_dim,
    }
    write_json(manifests_root / "train_manifest.json", {**manifest_payload, "trajectories": train_trajectories})
    write_json(manifests_root / "val_manifest.json", {**manifest_payload, "trajectories": val_trajectories})

    print(f"Dataset written to: {dataset_root}")
    print(f"Train trajectories: {len(train_trajectories)}")
    print(f"Val trajectories: {len(val_trajectories)}")


if __name__ == "__main__":
    main()
