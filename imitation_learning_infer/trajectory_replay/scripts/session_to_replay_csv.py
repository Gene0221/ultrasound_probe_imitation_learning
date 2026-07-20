#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "convert_session.yaml"


def load_config(path: str | Path | None) -> dict[str, Any]:
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        if path:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return payload


def config_get(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    payload = config.get(section, {})
    if isinstance(payload, dict) and key in payload:
        return payload[key]
    return default


def resolve_path(value: str | Path | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def resolve_session(config: dict[str, Any], session_arg: str | None) -> Path:
    if session_arg:
        session = resolve_path(session_arg, base=Path.cwd())
        if session is None:
            raise ValueError("Session path resolved to None.")
        return session

    value = config_get(config, "paths", "session") or config.get("session")
    if not value:
        raise ValueError("No session provided. Set paths.session in config or pass --session.")
    session = resolve_path(value)
    if session is None:
        raise ValueError("Session path resolved to None.")
    return session


def resolve_session_file(
    *,
    session: Path,
    explicit_path: str | None,
    configured_path: str | None,
    subdir: str,
    file_name: str,
) -> Path:
    path = resolve_path(explicit_path, base=Path.cwd()) if explicit_path else resolve_path(configured_path)
    if path is not None:
        return path
    return session / subdir / file_name


def resolve_output_path(
    *,
    session: Path,
    output_arg: str | None,
    output_cfg: str | None,
    output_subdir: str,
    output_file: str,
) -> Path:
    path = resolve_path(output_arg, base=Path.cwd()) if output_arg else resolve_path(output_cfg)
    if path is not None:
        return path
    return session / output_subdir / output_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(payload)
    return records


def normalize_quaternion(q: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return [v / n for v in q]


def quat_to_matrix(q: list[float]) -> list[list[float]]:
    x, y, z, w = normalize_quaternion(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def make_transform(translation: list[float], quaternion: list[float]) -> list[list[float]]:
    m = quat_to_matrix(quaternion)
    m[0][3], m[1][3], m[2][3] = translation
    return m


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def matrix_to_quat(m: list[list[float]]) -> list[float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return normalize_quaternion([qx, qy, qz, qw])


def identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def get_timestamp(record: dict[str, Any]) -> float:
    for key in ("host_timestamp_s", "curr_host_timestamp_s", "timestamp_s"):
        if key in record:
            return float(record[key])
    raise KeyError("Pose record has no timestamp field.")


def get_force_value(record: dict[str, Any]) -> float:
    prediction = record.get("prediction")
    if isinstance(prediction, dict):
        for key in ("Fz", "fz", "force_z"):
            if key in prediction:
                return float(prediction[key])
    values = record.get("predicted_values")
    if isinstance(values, list) and values:
        return float(values[0])
    return 0.0


def nearest_force(timestamp: float, force_records: list[dict[str, Any]], force_times: list[float], max_delta_s: float) -> float:
    if not force_records:
        return 0.0
    pos = bisect_left(force_times, timestamp)
    candidates = []
    if pos < len(force_times):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best = min(candidates, key=lambda idx: abs(force_times[idx] - timestamp))
    if abs(force_times[best] - timestamp) > max_delta_s:
        return 0.0
    return get_force_value(force_records[best])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert processed session JSONL files into Franka replay CSV.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to converter YAML config file. (default: config/convert_session.yaml)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session directory containing transformed_pose/ and predicted_force/. Overrides paths.session in config.",
    )
    parser.add_argument("--output", default=None, help="Output CSV path. Overrides replay.output_path in config.")
    parser.add_argument("--pose-file", default=None, help="Pose JSONL path. Overrides input.pose_file in config.")
    parser.add_argument("--force-file", default=None, help="Force JSONL path. Overrides input.force_file in config.")
    parser.add_argument(
        "--max-force-time-delta-s",
        type=float,
        default=None,
        help="Maximum timestamp delta used when matching force records. Overrides replay.max_force_time_delta_s.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    session_layout = config.get("session_layout", {})
    replay_cfg = config.get("replay", {})

    session = resolve_session(config, args.session)
    pose_path = resolve_session_file(
        session=session,
        explicit_path=args.pose_file,
        configured_path=config_get(config, "input", "pose_file"),
        subdir=str(session_layout.get("pose_subdir", "transformed_pose")),
        file_name=str(session_layout.get("pose_file", "flange_pose_deltas.jsonl")),
    )
    force_path = resolve_session_file(
        session=session,
        explicit_path=args.force_file,
        configured_path=config_get(config, "input", "force_file"),
        subdir=str(session_layout.get("force_subdir", "predicted_force")),
        file_name=str(session_layout.get("force_file", "predicted_force.jsonl")),
    )
    output_path = resolve_output_path(
        session=session,
        output_arg=args.output,
        output_cfg=config_get(config, "replay", "output_path"),
        output_subdir=str(replay_cfg.get("output_subdir", "franka_replay")),
        output_file=str(replay_cfg.get("output_file", "replay_trajectory.csv")),
    )
    max_force_time_delta_s = (
        args.max_force_time_delta_s
        if args.max_force_time_delta_s is not None
        else float(replay_cfg.get("max_force_time_delta_s", 0.05))
    )

    print(f"[INFO] Session: {session}")
    print(f"[INFO] Pose input: {pose_path}")
    print(f"[INFO] Force input: {force_path}")
    print(f"[INFO] Replay output: {output_path}")

    pose_records = load_jsonl(pose_path)
    force_records = load_jsonl(force_path) if force_path.exists() else []
    force_times = [get_timestamp(r) for r in force_records]
    if not pose_records:
        raise ValueError(f"No pose records found in {pose_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cumulative = identity()
    t0 = get_timestamp(pose_records[0])

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "x", "y", "z", "qx", "qy", "qz", "qw", "target_fz"])
        writer.writerow([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, nearest_force(t0, force_records, force_times, max_force_time_delta_s)])
        for record in pose_records:
            ts = get_timestamp(record)
            translation = [float(v) for v in record["delta_translation_xyz"]]
            quaternion = [float(v) for v in record["delta_quaternion_xyzw"]]
            cumulative = matmul(cumulative, make_transform(translation, quaternion))
            q = matrix_to_quat(cumulative)
            writer.writerow([
                f"{ts - t0:.6f}",
                f"{cumulative[0][3]:.9f}",
                f"{cumulative[1][3]:.9f}",
                f"{cumulative[2][3]:.9f}",
                f"{q[0]:.12f}",
                f"{q[1]:.12f}",
                f"{q[2]:.12f}",
                f"{q[3]:.12f}",
                f"{nearest_force(ts, force_records, force_times, max_force_time_delta_s):.6f}",
            ])

    print(f"[DONE] Wrote replay CSV: {output_path}")
    print(f"[INFO] Pose records: {len(pose_records)}, force records: {len(force_records)}")


if __name__ == "__main__":
    main()
