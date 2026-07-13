from __future__ import annotations

from bisect import bisect_left
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any

import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments
    def tqdm(iterable, **_: Any):
        return iterable


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def session_sort_key(path: Path, prefix: str) -> tuple[int, int | str]:
    suffix = path.name[len(prefix) :]
    if suffix.isdigit():
        return (0, int(suffix))
    match = re.search(r"(\d+)$", path.name)
    if match:
        return (1, int(match.group(1)))
    return (2, path.name)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    return records


def timestamp_of(record: dict[str, Any]) -> float:
    if "host_timestamp_s" in record:
        return float(record["host_timestamp_s"])
    if "curr_host_timestamp_s" in record:
        return float(record["curr_host_timestamp_s"])
    raise KeyError("Record has no host_timestamp_s or curr_host_timestamp_s.")


def nearest_index(timestamp_s: float, timestamps: list[float]) -> tuple[int | None, float]:
    if not timestamps:
        return None, float("inf")
    insert_index = bisect_left(timestamps, timestamp_s)
    candidates: list[int] = []
    if insert_index < len(timestamps):
        candidates.append(insert_index)
    if insert_index > 0:
        candidates.append(insert_index - 1)
    best_index = min(candidates, key=lambda idx: abs(timestamps[idx] - timestamp_s))
    return best_index, abs(timestamps[best_index] - timestamp_s)


def pose_delta(record: dict[str, Any]) -> list[float]:
    translation = [float(value) for value in record["delta_translation_xyz"]]
    quaternion = [float(value) for value in record["delta_quaternion_xyzw"]]
    if len(translation) != 3 or len(quaternion) != 4:
        raise ValueError("Pose delta must be 3D translation + 4D quaternion.")
    return [*translation, *quaternion]


def force_value(record: dict[str, Any]) -> float:
    if "predicted_values" in record and record["predicted_values"]:
        return float(record["predicted_values"][0])
    prediction = record.get("prediction")
    if isinstance(prediction, dict):
        for key in ("Fz", "force", "normal_force"):
            if key in prediction:
                return float(prediction[key])
        if prediction:
            return float(next(iter(prediction.values())))
    raise KeyError("Force record has no predicted_values[0] or prediction value.")


def ultrasound_image_path(session_dir: Path, layout: dict[str, Any], record: dict[str, Any]) -> Path:
    image_value = record.get("image") or record.get("image_path") or record.get("file")
    if not image_value:
        frame_index = int(record.get("frame_index", -1))
        if frame_index < 0:
            raise KeyError("Ultrasound record has no image path or frame_index.")
        image_value = f"frame_{frame_index:06d}.png"
    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path
    ultrasound_dir = session_dir / str(layout.get("ultrasound_subdir", "ultrasound"))
    if image_path.parts and image_path.parts[0] == str(layout.get("ultrasound_images_subdir", "images")):
        return ultrasound_dir / image_path
    return ultrasound_dir / str(layout.get("ultrasound_images_subdir", "images")) / image_path


def scan_sessions(session_root: Path, prefix: str) -> list[Path]:
    sessions = sorted(
        (path for path in session_root.iterdir() if path.is_dir() and path.name.startswith(prefix)),
        key=lambda path: session_sort_key(path, prefix),
    )
    if not sessions:
        raise FileNotFoundError(f"No {prefix}* sessions found under {session_root}")
    return sessions


def split_sessions(session_dirs: list[Path], config: dict[str, Any]) -> dict[str, list[Path]]:
    split_cfg = config["split"]
    rng = random.Random(int(split_cfg.get("seed", 42)))
    shuffled = list(session_dirs)
    rng.shuffle(shuffled)
    count = len(shuffled)
    train_count = int(count * float(split_cfg.get("train_ratio", 0.7)))
    val_count = int(count * float(split_cfg.get("val_ratio", 0.15)))
    if count >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        if train_count + val_count >= count:
            val_count = max(1, count - train_count - 1)
    test_count = count - train_count - val_count
    if count >= 3 and test_count <= 0:
        test_count = 1
        train_count = max(1, count - val_count - test_count)
    return {
        "train": sorted(shuffled[:train_count], key=lambda path: session_sort_key(path, config["input"].get("session_prefix", "session_"))),
        "val": sorted(shuffled[train_count : train_count + val_count], key=lambda path: session_sort_key(path, config["input"].get("session_prefix", "session_"))),
        "test": sorted(shuffled[train_count + val_count :], key=lambda path: session_sort_key(path, config["input"].get("session_prefix", "session_"))),
    }


def build_session_samples(
    session_dir: Path,
    config: dict[str, Any],
    *,
    include_force: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    layout = config["session_layout"]
    alignment = config["alignment"]
    horizon = int(config["dataset"].get("action_horizon", 20))

    ultrasound_path = session_dir / layout.get("ultrasound_subdir", "ultrasound") / layout.get("ultrasound_timestamps_file", "timestamps.jsonl")
    pose_path = session_dir / layout.get("pose_subdir", "transformed_pose") / layout.get("pose_file", "flange_pose_deltas.jsonl")
    force_path = session_dir / layout.get("force_subdir", "predicted_force") / layout.get("force_file", "predicted_force.jsonl")

    if not ultrasound_path.exists() or not pose_path.exists():
        return [], {"missing_required_files": 1}

    ultrasound_records = load_jsonl(ultrasound_path)
    pose_records = load_jsonl(pose_path)
    force_records = load_jsonl(force_path) if force_path.exists() else []

    pose_timestamps = [timestamp_of(record) for record in pose_records]
    force_timestamps = [timestamp_of(record) for record in force_records]
    max_pose_delta_s = float(alignment.get("max_pose_delta_s", 0.05))
    max_force_delta_s = float(alignment.get("max_force_delta_s", 0.05))

    samples: list[dict[str, Any]] = []
    stats = {
        "ultrasound_records": len(ultrasound_records),
        "pose_records": len(pose_records),
        "force_records": len(force_records),
        "skipped_pose_match": 0,
        "skipped_tail": 0,
        "skipped_force_match": 0,
        "missing_images": 0,
    }

    for sample_index, ultrasound_record in enumerate(ultrasound_records):
        timestamp_s = timestamp_of(ultrasound_record)
        pose_index, pose_delta_s = nearest_index(timestamp_s, pose_timestamps)
        if pose_index is None or pose_delta_s > max_pose_delta_s:
            stats["skipped_pose_match"] += 1
            continue
        future_start = pose_index + 1
        future_end = future_start + horizon
        if future_end > len(pose_records):
            stats["skipped_tail"] += 1
            continue

        image_source = ultrasound_image_path(session_dir, layout, ultrasound_record)
        if not image_source.exists():
            stats["missing_images"] += 1
            continue

        sample: dict[str, Any] = {
            "sample_id": f"{session_dir.name}_{sample_index:06d}",
            "source_session": session_dir.name,
            "source_image": str(image_source),
            "image": "",
            "timestamp_s": timestamp_s,
            "source_frame_index": ultrasound_record.get("frame_index"),
            "pose_anchor_index": int(pose_index),
            "pose_match_delta_s": float(pose_delta_s),
            "action_horizon": horizon,
            "action_chunk": [pose_delta(record) for record in pose_records[future_start:future_end]],
            "action_timestamps_s": [timestamp_of(record) for record in pose_records[future_start:future_end]],
        }

        if include_force:
            force_index, force_delta_s = nearest_index(timestamp_s, force_timestamps)
            if force_index is None or force_delta_s > max_force_delta_s:
                stats["skipped_force_match"] += 1
                continue
            sample["force"] = force_value(force_records[force_index])
            sample["force_match_delta_s"] = float(force_delta_s)
            sample["force_source_index"] = int(force_index)

        samples.append(sample)

    stats["samples"] = len(samples)
    return samples, stats


def copy_sample_images(samples: list[dict[str, Any]], image_dir: Path, *, desc: str) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for sample in tqdm(samples, desc=desc, unit="img", leave=False):
        source = Path(str(sample["source_image"]))
        extension = source.suffix or ".png"
        target_name = f"{sample['sample_id']}{extension}"
        target = image_dir / target_name
        shutil.copy2(source, target)
        sample["image"] = str(Path("images") / target_name)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_dataset(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    session_root = Path(str(config["input"]["session_root"])).resolve()
    session_prefix = str(config["input"].get("session_prefix", "session_"))
    dataset_root = Path(str(config["output"]["dataset_root"])).resolve()
    overwrite = bool(config["dataset"].get("overwrite", True))
    versions = list(config["dataset"].get("include_versions", ["without_force", "with_force"]))

    sessions = scan_sessions(session_root, session_prefix)
    split_map = split_sessions(sessions, config)
    summary: dict[str, Any] = {
        "config": config,
        "session_root": str(session_root),
        "dataset_root": str(dataset_root),
        "splits": {},
        "versions": {},
    }

    for version in tqdm(versions, desc="dataset versions", unit="version"):
        include_force = version == "with_force"
        version_root = dataset_root / version
        if version_root.exists() and overwrite:
            shutil.rmtree(version_root)
        version_summary = {"include_force": include_force, "splits": {}}

        for split_name, split_sessions_list in tqdm(split_map.items(), desc=f"{version} splits", unit="split", leave=False):
            split_summary = {"sessions": [], "num_samples": 0}
            for session_dir in tqdm(split_sessions_list, desc=f"{version}/{split_name} sessions", unit="session", leave=False):
                samples, stats = build_session_samples(session_dir, config, include_force=include_force)
                session_out = version_root / split_name / session_dir.name
                image_dir = session_out / "images"
                copy_sample_images(samples, image_dir, desc=f"copy {version}/{split_name}/{session_dir.name}")
                write_json(session_out / "samples.json", samples)
                write_json(session_out / "session_summary.json", stats)
                split_summary["sessions"].append(
                    {
                        "session_id": session_dir.name,
                        "output_dir": str(session_out),
                        **stats,
                    }
                )
                split_summary["num_samples"] += len(samples)
            version_summary["splits"][split_name] = split_summary

        write_json(version_root / "dataset_config.json", config)
        write_json(version_root / "dataset_summary.json", version_summary)
        summary["versions"][version] = version_summary

    summary["splits"] = {name: [path.name for path in paths] for name, paths in split_map.items()}
    write_json(dataset_root / "dataset_summary.json", summary)
    return summary

