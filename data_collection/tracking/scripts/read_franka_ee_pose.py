from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    import pylibfranka
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pylibfranka is required for this script. Install it before running."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class PoseSample:
    sample_index: int
    host_timestamp_s: float
    host_time_iso: str
    transform_base_ee: np.ndarray
    position_xyz: np.ndarray
    quaternion_xyzw: np.ndarray
    pose_source_field: str


class StopRequested(Exception):
    pass


class FrankaRobotAdapter:
    def __init__(self, robot_ip: str) -> None:
        self.robot_ip = robot_ip
        self.robot = self._connect(robot_ip)

    def _connect(self, robot_ip: str) -> Any:
        candidate_constructors = [
            ("Robot", lambda cls: cls(robot_ip)),
            ("Robot", lambda cls: cls(ip=robot_ip)),
            ("Robot", lambda cls: cls(robot_ip=robot_ip)),
            ("FrankaRobot", lambda cls: cls(robot_ip)),
            ("FrankaRobot", lambda cls: cls(ip=robot_ip)),
            ("FrankaRobot", lambda cls: cls(robot_ip=robot_ip)),
        ]

        for attr_name, factory in candidate_constructors:
            cls = getattr(pylibfranka, attr_name, None)
            if cls is None:
                continue
            try:
                return factory(cls)
            except TypeError:
                continue

        available = sorted(name for name in dir(pylibfranka) if not name.startswith("_"))
        raise RuntimeError(
            "Unable to construct a Franka robot from pylibfranka. "
            f"Available top-level symbols: {available}"
        )

    def read_state(self) -> Any:
        candidate_methods = [
            "read_once",
            "read_once_state",
            "get_state",
            "robot_state",
            "state",
            "read",
        ]

        for method_name in candidate_methods:
            method = getattr(self.robot, method_name, None)
            if method is None:
                continue
            try:
                return method()
            except TypeError:
                continue

        raise RuntimeError(
            "Unable to read robot state from pylibfranka robot instance. "
            "Tried methods: "
            + ", ".join(candidate_methods)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Franka end-effector pose and log adjacent-sample pose deltas."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    root = base_dir if base_dir is not None else PROJECT_ROOT
    return (root / path).resolve()


def load_config(path_value: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path_value)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return data


def now_iso8601() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"


def normalize_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    return quaternion_xyzw / norm


def rotation_matrix_to_quaternion_xyzw(rotation_matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation_matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        qz = 0.25 * s
    return normalize_quaternion_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def parse_transform_4x4(raw_value: Any) -> np.ndarray:
    raw = np.asarray(raw_value, dtype=np.float64).reshape(-1)
    if raw.size != 16:
        raise ValueError("End-effector transform must contain 16 values.")
    # Franka pose arrays are naturally reshaped as column-major homogeneous transforms.
    transform = np.reshape(raw, (4, 4), order="F")
    return transform


def get_pose_transform_and_field(
    robot_state: Any,
    preferred_field: str,
    fallback_fields: list[str],
) -> tuple[np.ndarray, str]:
    candidate_fields = [preferred_field] + [field for field in fallback_fields if field != preferred_field]

    for field_name in candidate_fields:
        if isinstance(robot_state, dict) and field_name in robot_state:
            return parse_transform_4x4(robot_state[field_name]), field_name
        if hasattr(robot_state, field_name):
            return parse_transform_4x4(getattr(robot_state, field_name)), field_name

    if isinstance(robot_state, dict):
        available = sorted(robot_state.keys())
    else:
        available = sorted(name for name in dir(robot_state) if not name.startswith("_"))
    raise RuntimeError(
        "Cannot find end-effector transform field in robot state. "
        f"Tried: {candidate_fields}. Available: {available}"
    )


def build_sample(
    sample_index: int,
    host_timestamp_s: float,
    host_time_iso: str,
    transform_base_ee: np.ndarray,
    pose_source_field: str,
) -> PoseSample:
    rotation = transform_base_ee[:3, :3]
    position = transform_base_ee[:3, 3].astype(np.float64).copy()
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)
    return PoseSample(
        sample_index=sample_index,
        host_timestamp_s=host_timestamp_s,
        host_time_iso=host_time_iso,
        transform_base_ee=transform_base_ee,
        position_xyz=position,
        quaternion_xyzw=quaternion,
        pose_source_field=pose_source_field,
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_path = resolve_path(args.config)

    robot_cfg = config.get("robot", {})
    sampling_cfg = config.get("sampling", {})
    output_cfg = config.get("output", {})
    recording_cfg = config.get("recording", {})

    robot_ip = str(robot_cfg.get("ip", "")).strip()
    if not robot_ip:
        raise ValueError("robot.ip must be set in the config file.")

    preferred_field = str(robot_cfg.get("pose_source_field", "O_T_EE")).strip() or "O_T_EE"
    fallback_fields = [str(item).strip() for item in robot_cfg.get("fallback_pose_source_fields", []) if str(item).strip()]
    target_hz = float(sampling_cfg.get("target_hz", 30.0))
    if target_hz <= 0.0:
        raise ValueError("sampling.target_hz must be positive.")
    dt = 1.0 / target_hz

    max_samples_value = sampling_cfg.get("max_samples", None)
    max_samples = None if max_samples_value in (None, "", "null") else int(max_samples_value)

    output_root = resolve_path(str(output_cfg.get("output_root", "output")), config_path.parent.parent)
    jsonl_file_name = str(output_cfg.get("jsonl_file_name", "franka_ee_pose_deltas.jsonl"))
    summary_file_name = str(output_cfg.get("summary_file_name", "summary.json"))
    write_jsonl = bool(output_cfg.get("write_jsonl", True))
    emit_stdout_records = bool(output_cfg.get("emit_stdout_records", False))

    include_absolute_pose = bool(recording_cfg.get("include_absolute_pose", True))
    include_pose_source_field = bool(recording_cfg.get("include_pose_source_field", True))
    include_robot_ip = bool(recording_cfg.get("include_robot_ip", True))

    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / jsonl_file_name
    summary_path = output_root / summary_file_name

    stop_requested = False

    def handle_sigint(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"[INFO] Connecting to Franka robot at {robot_ip}")
    adapter = FrankaRobotAdapter(robot_ip)
    print(f"[INFO] Logging pose deltas at {target_hz:.2f} Hz")
    print(f"[INFO] Output directory: {output_root}")
    print("[INFO] Press Ctrl+C to stop.")

    previous_sample: Optional[PoseSample] = None
    records_logged = 0
    sample_index = 0
    first_timestamp_s: Optional[float] = None
    last_timestamp_s: Optional[float] = None
    pose_source_field_used: Optional[str] = None

    while not stop_requested:
        loop_start = time.perf_counter()
        host_timestamp_s = time.time()
        host_time_iso = now_iso8601()
        robot_state = adapter.read_state()
        transform_base_ee, pose_source_field = get_pose_transform_and_field(
            robot_state,
            preferred_field=preferred_field,
            fallback_fields=fallback_fields,
        )

        sample_index += 1
        current_sample = build_sample(
            sample_index=sample_index,
            host_timestamp_s=host_timestamp_s,
            host_time_iso=host_time_iso,
            transform_base_ee=transform_base_ee,
            pose_source_field=pose_source_field,
        )

        if first_timestamp_s is None:
            first_timestamp_s = current_sample.host_timestamp_s
        last_timestamp_s = current_sample.host_timestamp_s
        pose_source_field_used = current_sample.pose_source_field

        if previous_sample is not None:
            delta_transform = invert_transform(previous_sample.transform_base_ee) @ current_sample.transform_base_ee
            delta_translation = delta_transform[:3, 3].astype(np.float64)
            delta_quaternion = rotation_matrix_to_quaternion_xyzw(delta_transform[:3, :3])

            record: dict[str, Any] = {
                "sample_index": current_sample.sample_index,
                "valid": True,
                "prev_host_timestamp_s": previous_sample.host_timestamp_s,
                "curr_host_timestamp_s": current_sample.host_timestamp_s,
                "prev_host_time_iso": previous_sample.host_time_iso,
                "curr_host_time_iso": current_sample.host_time_iso,
                "delta_dt_s": current_sample.host_timestamp_s - previous_sample.host_timestamp_s,
                "delta_transform_prev_to_curr": delta_transform.tolist(),
                "delta_translation_xyz": delta_translation.tolist(),
                "delta_quaternion_xyzw": delta_quaternion.tolist(),
            }

            if include_absolute_pose:
                record["prev_position_xyz"] = previous_sample.position_xyz.tolist()
                record["prev_quaternion_xyzw"] = previous_sample.quaternion_xyzw.tolist()
                record["curr_position_xyz"] = current_sample.position_xyz.tolist()
                record["curr_quaternion_xyzw"] = current_sample.quaternion_xyzw.tolist()

            if include_pose_source_field:
                record["pose_source_field"] = current_sample.pose_source_field

            if include_robot_ip:
                record["robot_ip"] = robot_ip

            if write_jsonl:
                append_jsonl(jsonl_path, record)
                records_logged += 1
            if emit_stdout_records:
                print(json.dumps(record), flush=True)
            else:
                print(
                    "[POSE] "
                    f"idx={current_sample.sample_index} "
                    f"dt={record['delta_dt_s']:.4f}s "
                    f"dxyz=({delta_translation[0]:+.4f}, {delta_translation[1]:+.4f}, {delta_translation[2]:+.4f}) "
                    f"dq=({delta_quaternion[0]:+.5f}, {delta_quaternion[1]:+.5f}, {delta_quaternion[2]:+.5f}, {delta_quaternion[3]:+.5f})"
                )

        previous_sample = current_sample

        if max_samples is not None and sample_index >= max_samples:
            break

        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, dt - elapsed))

    summary = {
        "output_root": str(output_root),
        "jsonl_path": str(jsonl_path),
        "robot_ip": robot_ip,
        "target_hz": target_hz,
        "records_logged": records_logged,
        "samples_read": sample_index,
        "pose_source_field": pose_source_field_used,
        "first_host_timestamp_s": first_timestamp_s,
        "last_host_timestamp_s": last_timestamp_s,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
