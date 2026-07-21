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


def resolve_output_dir(output_dir_arg: str | None, output_dir_cfg: str | None) -> Path:
    value = output_dir_arg or output_dir_cfg
    if not value:
        raise ValueError("No replay output directory provided. Set replay.output_dir in config or pass --output-dir.")
    base = Path.cwd() if output_dir_arg else PROJECT_ROOT
    output_dir = resolve_path(value, base=base)
    if output_dir is None:
        raise ValueError("Replay output directory resolved to None.")
    return output_dir


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


def is_identity_delta(translation: list[float], quaternion: list[float], tol: float = 1e-9) -> bool:
    q = normalize_quaternion(quaternion)
    return (
        all(abs(v) <= tol for v in translation)
        and abs(q[0]) <= tol
        and abs(q[1]) <= tol
        and abs(q[2]) <= tol
        and abs(q[3] - 1.0) <= tol
    )


def get_timestamp(record: dict[str, Any]) -> float:
    for key in ("host_timestamp_s", "curr_host_timestamp_s", "timestamp_s"):
        if key in record:
            return float(record[key])
    raise KeyError("Pose record has no timestamp field.")


def get_start_timestamp(record: dict[str, Any]) -> float:
    if "prev_host_timestamp_s" in record:
        return float(record["prev_host_timestamp_s"])
    return get_timestamp(record)


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


def lowpass_series(times: list[float], values: list[float], cutoff_hz: float) -> list[float]:
    if cutoff_hz <= 0.0:
        raise ValueError("filter.cutoff_hz must be positive.")
    if not values:
        return []
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    filtered = [values[0]]
    for idx in range(1, len(values)):
        dt = max(times[idx] - times[idx - 1], 1e-9)
        alpha = dt / (tau + dt)
        filtered.append(filtered[-1] + alpha * (values[idx] - filtered[-1]))
    return filtered


def lowpass_zero_phase(times: list[float], values: list[float], cutoff_hz: float) -> list[float]:
    forward = lowpass_series(times, values, cutoff_hz)
    reversed_times = [times[-1] - t for t in reversed(times)]
    backward = list(reversed(lowpass_series(reversed_times, list(reversed(forward)), cutoff_hz)))
    backward[0] = values[0]
    backward[-1] = values[-1]
    return backward


def apply_position_lowpass(rows: list[dict[str, float]], filter_cfg: dict[str, Any]) -> list[dict[str, float]]:
    if not bool(filter_cfg.get("enabled", False)):
        return rows
    cutoff_hz = float(filter_cfg.get("cutoff_hz", 1.0))
    zero_phase = bool(filter_cfg.get("zero_phase", True))
    times = [row["time_s"] for row in rows]
    output = [dict(row) for row in rows]
    for key in ("x", "y", "z"):
        values = [row[key] for row in rows]
        filtered = (
            lowpass_zero_phase(times, values, cutoff_hz)
            if zero_phase
            else lowpass_series(times, values, cutoff_hz)
        )
        for row, value in zip(output, filtered):
            row[key] = value
    return output


def write_replay_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "x", "y", "z", "qx", "qy", "qz", "qw", "target_fz"])
        for row in rows:
            writer.writerow([
                f"{row['time_s']:.6f}",
                f"{row['x']:.9f}",
                f"{row['y']:.9f}",
                f"{row['z']:.9f}",
                f"{row['qx']:.12f}",
                f"{row['qy']:.12f}",
                f"{row['qz']:.12f}",
                f"{row['qw']:.12f}",
                f"{row['target_fz']:.6f}",
            ])


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
    parser.add_argument("--output-dir", default=None, help="Directory for replay CSV files. Overrides replay.output_dir in config.")
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
    filter_cfg = config.get("filter", {})

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
    output_dir = resolve_output_dir(args.output_dir, config_get(config, "replay", "output_dir"))
    output_path = output_dir / str(replay_cfg.get("output_file", "replay_trajectory.csv"))
    raw_output_path = output_dir / str(replay_cfg.get("raw_output_file", "replay_trajectory_raw.csv"))
    max_force_time_delta_s = (
        args.max_force_time_delta_s
        if args.max_force_time_delta_s is not None
        else float(replay_cfg.get("max_force_time_delta_s", 0.05))
    )

    print(f"[INFO] Session: {session}")
    print(f"[INFO] Pose input: {pose_path}")
    print(f"[INFO] Force input: {force_path}")
    print(f"[INFO] Replay output dir: {output_dir}")
    print(f"[INFO] Replay output: {output_path}")

    pose_records = load_jsonl(pose_path)
    force_records = load_jsonl(force_path) if force_path.exists() else []
    force_times = [get_timestamp(r) for r in force_records]
    if not pose_records:
        raise ValueError(f"No pose records found in {pose_path}")

    cumulative = identity()
    t0 = get_start_timestamp(pose_records[0])
    raw_rows: list[dict[str, float]] = [
        {
            "time_s": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "target_fz": nearest_force(t0, force_records, force_times, max_force_time_delta_s),
        }
    ]
    last_time_s = 0.0
    skipped_duplicate_identity = 0

    for record in pose_records:
        ts = get_timestamp(record)
        time_s = ts - t0
        translation = [float(v) for v in record["delta_translation_xyz"]]
        quaternion = [float(v) for v in record["delta_quaternion_xyzw"]]
        if time_s <= last_time_s:
            if is_identity_delta(translation, quaternion):
                skipped_duplicate_identity += 1
                continue
            raise ValueError(
                f"Non-increasing trajectory time {time_s:.9f}s after {last_time_s:.9f}s. "
                "Check prev_host_timestamp_s/curr_host_timestamp_s in the pose JSONL."
            )
        cumulative = matmul(cumulative, make_transform(translation, quaternion))
        q = matrix_to_quat(cumulative)
        raw_rows.append(
            {
                "time_s": time_s,
                "x": cumulative[0][3],
                "y": cumulative[1][3],
                "z": cumulative[2][3],
                "qx": q[0],
                "qy": q[1],
                "qz": q[2],
                "qw": q[3],
                "target_fz": nearest_force(ts, force_records, force_times, max_force_time_delta_s),
            }
        )
        last_time_s = time_s

    if bool(replay_cfg.get("write_raw_copy", True)):
        write_replay_csv(raw_output_path, raw_rows)
        print(f"[INFO] Wrote raw replay CSV: {raw_output_path}")

    output_rows = apply_position_lowpass(raw_rows, filter_cfg)
    write_replay_csv(output_path, output_rows)

    print(f"[DONE] Wrote replay CSV: {output_path}")
    print(f"[INFO] Output rows: {len(output_rows)} (raw rows: {len(raw_rows)})")
    if filter_cfg.get("enabled", False):
        print(f"[INFO] Position low-pass filter enabled: {filter_cfg}")
    print(f"[INFO] Pose records: {len(pose_records)}, force records: {len(force_records)}")
    if skipped_duplicate_identity:
        print(f"[INFO] Skipped duplicate identity records at non-increasing timestamps: {skipped_duplicate_identity}")


if __name__ == "__main__":
    main()


