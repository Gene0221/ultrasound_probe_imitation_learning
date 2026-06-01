from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyrealsense2 is required for this script. Install librealsense/pyrealsense2 first."
) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class MotionSample:
    stream_name: str
    xyz: np.ndarray
    device_timestamp_ms: float
    frame_number: int
    host_timestamp_s: float


@dataclass
class OrientationState:
    pitch_deg: float
    roll_deg: float
    tilt_deg: float
    accel_xyz: np.ndarray
    gyro_xyz: np.ndarray
    device_timestamp_ms: float
    frame_number: int
    host_timestamp_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log gravity-driven pitch/roll from Intel RealSense D435i IMU."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config file.")
    parser.add_argument("--list-devices", action="store_true", help="List detected RealSense devices and exit.")
    stdout_group = parser.add_mutually_exclusive_group()
    stdout_group.add_argument(
        "--print-stdout",
        dest="print_stdout",
        action="store_true",
        default=None,
        help="Print each emitted IMU record to stdout.",
    )
    stdout_group.add_argument(
        "--no-print-stdout",
        dest="print_stdout",
        action="store_false",
        help="Do not print emitted IMU records to stdout.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    root = base_dir if base_dir is not None else PROJECT_ROOT
    return (root / path).resolve()


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml_or_json(config_path)


def capture_host_timestamp_s() -> float:
    return time.time()


def list_realsense_devices() -> list[dict[str, str]]:
    devices_info: list[dict[str, str]] = []
    for device in rs.context().query_devices():
        name = device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else "unknown"
        serial = device.get_info(rs.camera_info.serial_number) if device.supports(rs.camera_info.serial_number) else ""
        firmware = device.get_info(rs.camera_info.firmware_version) if device.supports(rs.camera_info.firmware_version) else ""
        if not serial:
            continue
        devices_info.append({"name": name, "serial_no": serial, "firmware": firmware})
    return devices_info


def get_connected_device_by_serial(serial_no: str) -> rs.device | None:
    for device in rs.context().query_devices():
        if not device.supports(rs.camera_info.serial_number):
            continue
        if device.get_info(rs.camera_info.serial_number) == serial_no:
            return device
    return None


def get_motion_profile_summary(device: rs.device) -> dict[str, list[int]]:
    summary: dict[str, set[int]] = {"accel": set(), "gyro": set()}
    for sensor in device.query_sensors():
        try:
            profiles = sensor.get_stream_profiles()
        except Exception:
            continue
        for profile in profiles:
            stream_type = profile.stream_type()
            fps = int(profile.fps())
            if stream_type == rs.stream.accel:
                summary["accel"].add(fps)
            elif stream_type == rs.stream.gyro:
                summary["gyro"].add(fps)
    return {
        "accel": sorted(summary["accel"]),
        "gyro": sorted(summary["gyro"]),
    }


def select_device_serial(config: dict[str, Any]) -> str | None:
    serial_no = config.get("device", {}).get("serial_no")
    if serial_no is None:
        return None
    return str(serial_no).strip() or None


def make_pipeline(config: dict[str, Any]) -> tuple[rs.pipeline, rs.pipeline_profile]:
    imu_cfg = config.get("imu", {})
    accel_hz = int(imu_cfg.get("accel_hz", 250))
    gyro_hz = int(imu_cfg.get("gyro_hz", 200))

    pipeline = rs.pipeline()
    rs_config = rs.config()

    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError(
            "No RealSense devices detected. Check USB connection, power, and librealsense driver setup."
        )

    serial_no = select_device_serial(config)
    if serial_no:
        selected_device = get_connected_device_by_serial(serial_no)
        if selected_device is None:
            known_serials = ", ".join(device["serial_no"] for device in devices)
            raise RuntimeError(
                f"Configured serial_no '{serial_no}' was not found. Connected serial numbers: {known_serials}"
            )
        rs_config.enable_device(serial_no)
    else:
        selected_device = get_connected_device_by_serial(devices[0]["serial_no"])

    rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_hz)
    rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_hz)
    try:
        profile = pipeline.start(rs_config)
    except RuntimeError as exc:
        device_name = "unknown"
        device_serial = serial_no or "auto-select"
        profile_summary = {"accel": [], "gyro": []}
        if selected_device is not None:
            if selected_device.supports(rs.camera_info.name):
                device_name = selected_device.get_info(rs.camera_info.name)
            if selected_device.supports(rs.camera_info.serial_number):
                device_serial = selected_device.get_info(rs.camera_info.serial_number)
            profile_summary = get_motion_profile_summary(selected_device)
        raise RuntimeError(
            "Failed to start IMU streams for the selected RealSense device. "
            f"device_name={device_name}, serial_no={device_serial}, "
            f"requested_accel_hz={accel_hz}, requested_gyro_hz={gyro_hz}, "
            f"supported_accel_hz={profile_summary['accel']}, supported_gyro_hz={profile_summary['gyro']}. "
            "This usually means the selected device is not a D435i, the serial number is wrong, "
            "or the requested IMU profiles are unavailable."
        ) from exc
    return pipeline, profile


def motion_sample_from_frame(frame: rs.frame) -> MotionSample:
    motion = frame.as_motion_frame().get_motion_data()
    stream_name = frame.profile.stream_name().lower()
    return MotionSample(
        stream_name=stream_name,
        xyz=np.array([motion.x, motion.y, motion.z], dtype=np.float64),
        device_timestamp_ms=float(frame.get_timestamp()),
        frame_number=int(frame.get_frame_number()),
        host_timestamp_s=capture_host_timestamp_s(),
    )


def accel_to_pitch_roll_deg(accel_xyz: np.ndarray) -> tuple[float, float]:
    ax, ay, az = [float(v) for v in accel_xyz]
    pitch_rad = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    roll_rad = math.atan2(ay, az)
    return math.degrees(pitch_rad), math.degrees(roll_rad)


def compute_tilt_deg(accel_xyz: np.ndarray) -> float:
    accel_norm = float(np.linalg.norm(accel_xyz))
    if accel_norm <= 1e-9:
        return 0.0
    cos_theta = float(np.clip(accel_xyz[2] / accel_norm, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def compute_relative_tilt_deg(reference_accel_xyz: np.ndarray, current_accel_xyz: np.ndarray) -> float:
    ref_norm = float(np.linalg.norm(reference_accel_xyz))
    cur_norm = float(np.linalg.norm(current_accel_xyz))
    if ref_norm <= 1e-9 or cur_norm <= 1e-9:
        return 0.0
    ref_unit = reference_accel_xyz / ref_norm
    cur_unit = current_accel_xyz / cur_norm
    cosine = float(np.clip(np.dot(ref_unit, cur_unit), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def extract_motion_samples(frames: rs.composite_frame) -> list[MotionSample]:
    samples: list[MotionSample] = []
    accel_frame = frames.first_or_default(rs.stream.accel)
    gyro_frame = frames.first_or_default(rs.stream.gyro)

    if accel_frame and accel_frame.is_motion_frame():
        samples.append(motion_sample_from_frame(accel_frame))
    if gyro_frame and gyro_frame.is_motion_frame():
        samples.append(motion_sample_from_frame(gyro_frame))
    return samples


def wait_for_motion_samples(
    pipeline: rs.pipeline,
    timeout_ms: int = 5000,
) -> list[MotionSample]:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms)
        samples = extract_motion_samples(frames)
        if samples:
            return samples


def calibrate_gyro_bias_and_initial_orientation(
    pipeline: rs.pipeline,
    stationary_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    deadline = time.monotonic() + max(0.1, stationary_duration_s)
    gyro_samples: list[np.ndarray] = []
    accel_samples: list[np.ndarray] = []

    while time.monotonic() < deadline:
        for sample in wait_for_motion_samples(pipeline):
            if "gyro" in sample.stream_name:
                gyro_samples.append(sample.xyz)
            elif "accel" in sample.stream_name:
                accel_samples.append(sample.xyz)

    if not gyro_samples:
        raise RuntimeError("No gyro samples collected during calibration.")
    if not accel_samples:
        raise RuntimeError("No accel samples collected during calibration.")

    gyro_bias = np.mean(np.stack(gyro_samples, axis=0), axis=0)
    accel_mean = np.mean(np.stack(accel_samples, axis=0), axis=0)
    return gyro_bias, accel_mean


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def ensure_output_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    output_cfg = config.get("output", {})
    output_root = resolve_path(str(output_cfg.get("output_root", "output")))
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / str(output_cfg.get("jsonl_file_name", "imu_pitch_roll.jsonl"))
    summary_path = output_root / str(output_cfg.get("summary_file_name", "summary.json"))
    config_copy_path = output_root / "resolved_config.json"
    return output_root, jsonl_path, summary_path


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_record(
    state: OrientationState,
    recording_cfg: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "host_timestamp_s": state.host_timestamp_s,
        "pitch_deg": state.pitch_deg,
        "roll_deg": state.roll_deg,
    }
    if recording_cfg.get("include_tilt_deg", True):
        payload["tilt_deg"] = state.tilt_deg
    if recording_cfg.get("include_device_timestamp_ms", True):
        payload["device_timestamp_ms"] = state.device_timestamp_ms
    if recording_cfg.get("include_frame_number", True):
        payload["frame_number"] = state.frame_number
    if recording_cfg.get("include_raw_accel_xyz", False):
        payload["accel_xyz"] = state.accel_xyz.tolist()
    if recording_cfg.get("include_raw_gyro_xyz", False):
        payload["gyro_xyz"] = state.gyro_xyz.tolist()
    return payload


def run_logger(config: dict[str, Any], print_stdout_override: bool | None = None) -> None:
    calibration_cfg = config.get("calibration", {})
    filter_cfg = config.get("filter", {})
    sampling_cfg = config.get("sampling", {})
    output_cfg = config.get("output", {})
    recording_cfg = config.get("recording", {})

    output_root, jsonl_path, summary_path = ensure_output_paths(config)
    pipeline, profile = make_pipeline(config)

    stop_requested = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[INFO] Received signal {signum}; stopping logger...", file=sys.stderr)

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    write_jsonl = bool(output_cfg.get("write_jsonl", True))
    print_stdout = bool(output_cfg.get("print_stdout", True))
    if print_stdout_override is not None:
        print_stdout = bool(print_stdout_override)
    output_hz = float(sampling_cfg.get("output_hz", 100.0))
    max_samples = sampling_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    accel_low_pass_alpha = float(filter_cfg.get("accel_low_pass_alpha", 0.9))
    stationary_duration_s = float(calibration_cfg.get("stationary_duration_s", 2.0))

    profile_device = profile.get_device()
    device_name = profile_device.get_info(rs.camera_info.name) if profile_device.supports(rs.camera_info.name) else "unknown"
    device_serial = profile_device.get_info(rs.camera_info.serial_number) if profile_device.supports(rs.camera_info.serial_number) else ""

    first_host_timestamp_s: float | None = None
    last_host_timestamp_s: float | None = None
    latest_accel_xyz = np.zeros(3, dtype=np.float64)
    filtered_accel_xyz = np.zeros(3, dtype=np.float64)
    latest_gyro_xyz = np.zeros(3, dtype=np.float64)
    sample_count = 0
    last_emit_host_timestamp_s = 0.0

    try:
        print(
            f"[INFO] Connected device: {device_name} serial={device_serial or 'unknown'}",
            file=sys.stderr,
        )
        print(
            f"[INFO] Hold the device still for {stationary_duration_s:.2f} s to calibrate zero pose for gravity-driven pitch/roll.",
            file=sys.stderr,
        )
        gyro_bias, accel_mean = calibrate_gyro_bias_and_initial_orientation(
            pipeline,
            stationary_duration_s=stationary_duration_s,
        )
        latest_accel_xyz = accel_mean.copy()
        filtered_accel_xyz = accel_mean.copy()
        pitch_zero_deg, roll_zero_deg = accel_to_pitch_roll_deg(accel_mean)
        pitch_deg = 0.0
        roll_deg = 0.0
        tilt_deg = 0.0
        print(
            "[INFO] Calibration complete. "
            f"gyro_bias={gyro_bias.tolist()} pitch0={pitch_zero_deg:.3f} roll0={roll_zero_deg:.3f}",
            file=sys.stderr,
        )

        while not stop_requested:
            for sample in wait_for_motion_samples(pipeline):
                if "gyro" in sample.stream_name:
                    corrected_gyro = sample.xyz - gyro_bias
                    latest_gyro_xyz = corrected_gyro
                    continue

                if "accel" not in sample.stream_name:
                    continue

                latest_accel_xyz = sample.xyz
                filtered_accel_xyz = (
                    accel_low_pass_alpha * filtered_accel_xyz
                    + (1.0 - accel_low_pass_alpha) * latest_accel_xyz
                )
                gravity_pitch_deg, gravity_roll_deg = accel_to_pitch_roll_deg(filtered_accel_xyz)
                pitch_deg = gravity_pitch_deg - pitch_zero_deg
                roll_deg = gravity_roll_deg - roll_zero_deg
                tilt_deg = compute_relative_tilt_deg(accel_mean, filtered_accel_xyz)

                min_emit_interval_s = 0.0 if output_hz <= 0.0 else 1.0 / output_hz
                if sample.host_timestamp_s - last_emit_host_timestamp_s < min_emit_interval_s:
                    continue

                state = OrientationState(
                    pitch_deg=float(pitch_deg),
                    roll_deg=float(roll_deg),
                    tilt_deg=float(tilt_deg),
                    accel_xyz=latest_accel_xyz.copy(),
                    gyro_xyz=latest_gyro_xyz.copy(),
                    device_timestamp_ms=sample.device_timestamp_ms,
                    frame_number=sample.frame_number,
                    host_timestamp_s=sample.host_timestamp_s,
                )
                record = build_record(state, recording_cfg)

                if write_jsonl:
                    append_jsonl(jsonl_path, record)
                if print_stdout:
                    print(json.dumps(record, ensure_ascii=True), flush=True)

                if first_host_timestamp_s is None:
                    first_host_timestamp_s = sample.host_timestamp_s
                last_host_timestamp_s = sample.host_timestamp_s
                last_emit_host_timestamp_s = sample.host_timestamp_s
                sample_count += 1

                if max_samples is not None and sample_count >= max_samples:
                    stop_requested = True
                    break
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

        summary_payload = {
            "device_name": device_name,
            "device_serial_no": device_serial,
            "output_root": str(output_root),
            "jsonl_path": str(jsonl_path),
            "first_host_timestamp_s": first_host_timestamp_s,
            "last_host_timestamp_s": last_host_timestamp_s,
            "num_records": sample_count,
            "config": config,
        }
        write_summary(summary_path, summary_payload)
        print(f"[INFO] Wrote summary: {summary_path}", file=sys.stderr)
        if write_jsonl:
            print(f"[INFO] Wrote JSONL: {jsonl_path}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.list_devices:
        payload = {"devices": list_realsense_devices()}
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    config = load_config(args.config)
    run_logger(config, print_stdout_override=args.print_stdout)


if __name__ == "__main__":
    main()
