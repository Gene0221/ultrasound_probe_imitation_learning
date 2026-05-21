from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyrealsense2 is required for this script. Install librealsense/pyrealsense2 first."
    ) from exc

try:
    from pupil_apriltags import Detector
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pupil_apriltags is required for AprilTag tracking. Install it before running this script."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "vision" / "apriltag_tracking.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "vision" / "output"
@dataclass
class FrameRecord:
    image: np.ndarray
    device_timestamp_ms: float
    host_timestamp_s: float
    frame_number: int
    camera_label: str


@dataclass
class DetectionPose:
    tag_id: int
    transform_camera_tag: np.ndarray
    source_camera: str
    center_xy: tuple[float, float]
    corners_xy: np.ndarray
    decision_margin: float
    hamming: int


class FrameBuffer:
    def __init__(self, maxlen: int) -> None:
        self._frames: deque[FrameRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: FrameRecord) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self) -> list[FrameRecord]:
        with self._lock:
            return list(self._frames)

    def latest(self) -> Optional[FrameRecord]:
        with self._lock:
            return self._frames[-1] if self._frames else None


class CameraWorker(threading.Thread):
    def __init__(
        self,
        serial_no: str,
        camera_label: str,
        width: int,
        height: int,
        fps: int,
        buffer_size: int,
        warmup_frames: int,
    ) -> None:
        super().__init__(daemon=True)
        self.serial_no = serial_no
        self.camera_label = camera_label
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.buffer = FrameBuffer(buffer_size)
        self._pipeline = rs.pipeline()
        self._stop_event = threading.Event()
        self._started_ok = threading.Event()
        self._startup_error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            config = rs.config()
            config.enable_device(self.serial_no)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self._pipeline.start(config)
            self._started_ok.set()

            warmup_done = 0
            while not self._stop_event.is_set():
                try:
                    frames = self._pipeline.wait_for_frames(5000)
                except RuntimeError:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                if warmup_done < self.warmup_frames:
                    warmup_done += 1
                    continue

                self.buffer.push(
                    FrameRecord(
                        image=np.asanyarray(color_frame.get_data()).copy(),
                        device_timestamp_ms=float(color_frame.get_timestamp()),
                        host_timestamp_s=time.perf_counter(),
                        frame_number=int(color_frame.get_frame_number()),
                        camera_label=self.camera_label,
                    )
                )
        except BaseException as exc:  # pragma: no cover
            self._startup_error = exc
            self._started_ok.set()
        finally:
            try:
                self._pipeline.stop()
            except Exception:
                pass

    def wait_until_started(self, timeout_s: float) -> None:
        ok = self._started_ok.wait(timeout=timeout_s)
        if not ok:
            raise TimeoutError(f"{self.camera_label} did not start within {timeout_s:.1f} seconds.")
        if self._startup_error is not None:
            raise RuntimeError(f"{self.camera_label} failed to start: {self._startup_error}") from self._startup_error

    def stop(self) -> None:
        self._stop_event.set()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track per-tag pose deltas between adjacent frames using dual D435i RGB streams and AprilTag."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config file.")
    parser.add_argument("--list-devices", action="store_true", help="Only list detected RealSense devices and exit.")
    return parser.parse_args()


def resolve_path(path_value: str | Path, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    root = base_dir if base_dir is not None else PROJECT_ROOT
    return (root / path).resolve()


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in file: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml_or_json(config_path)


def prompt_session_name() -> str:
    return input("Enter session name: ").strip()


def parse_intrinsics_payload(payload: dict[str, Any], file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if "camera_matrix" in payload and "dist_coeffs" in payload:
        camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)
        return camera_matrix, dist_coeffs.reshape(-1, 1)
    if "cam0" in payload:
        camera_matrix = np.array(payload["cam0"]["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(payload["cam0"]["dist_coeffs"], dtype=np.float64)
        return camera_matrix, dist_coeffs.reshape(-1, 1)
    raise ValueError(f"Unsupported intrinsics file format: {file_path}")


def load_intrinsics(path_value: str | Path, base_dir: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    path = resolve_path(path_value, base_dir)
    payload = load_yaml_or_json(path)
    camera_matrix, dist_coeffs = parse_intrinsics_payload(payload, path)
    return path, camera_matrix, dist_coeffs


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation_matrix)
    if trace > 0:
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
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def select_frame_closest_to_time(frames: list[FrameRecord], target_host_time_s: float) -> Optional[FrameRecord]:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.host_timestamp_s - target_host_time_s))


def select_best_pair(buffer_a: FrameBuffer, buffer_b: FrameBuffer) -> tuple[Optional[FrameRecord], Optional[FrameRecord], Optional[float]]:
    latest_a = buffer_a.latest()
    if latest_a is None:
        return None, None, None
    best_b = select_frame_closest_to_time(buffer_b.snapshot(), latest_a.host_timestamp_s)
    if best_b is None:
        return latest_a, None, None
    delta_ms = abs(latest_a.host_timestamp_s - best_b.host_timestamp_s) * 1000.0
    return latest_a, best_b, delta_ms


def build_detector(config: dict[str, Any]) -> Detector:
    return Detector(
        families=str(config.get("family", "tag36h11")),
        nthreads=int(config.get("nthreads", 2)),
        quad_decimate=float(config.get("quad_decimate", 1.0)),
        quad_sigma=float(config.get("quad_sigma", 0.0)),
        refine_edges=int(config.get("refine_edges", 1)),
        decode_sharpening=float(config.get("decode_sharpening", 0.25)),
    )


def detect_tag_poses(
    detector: Detector,
    frame: FrameRecord,
    camera_matrix: np.ndarray,
    tag_size_m: float,
    tracked_tag_ids: set[int],
    allowed_hamming: int,
) -> dict[int, DetectionPose]:
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size_m)

    poses: dict[int, DetectionPose] = {}
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id not in tracked_tag_ids:
            continue
        if int(detection.hamming) > allowed_hamming:
            continue
        rotation = np.array(detection.pose_R, dtype=np.float64)
        translation = np.array(detection.pose_t, dtype=np.float64).reshape(3)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        poses[tag_id] = DetectionPose(
            tag_id=tag_id,
            transform_camera_tag=transform,
            source_camera=frame.camera_label,
            center_xy=(float(detection.center[0]), float(detection.center[1])),
            corners_xy=np.array(detection.corners, dtype=np.float64),
            decision_margin=float(detection.decision_margin),
            hamming=int(detection.hamming),
        )
    return poses


def choose_preferred_pose(
    tag_id: int,
    detections_a: dict[int, DetectionPose],
    detections_b: dict[int, DetectionPose],
) -> Optional[DetectionPose]:
    if tag_id in detections_a:
        return detections_a[tag_id]
    return detections_b.get(tag_id)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_detections(image: np.ndarray, detections: dict[int, DetectionPose], title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    for detection in detections.values():
        corners = detection.corners_xy.astype(int)
        cv2.polylines(canvas, [corners], True, (0, 255, 255), 2)
        center = (int(detection.center_xy[0]), int(detection.center_xy[1]))
        cv2.circle(canvas, center, 4, (0, 0, 255), -1)
        label = f"id={detection.tag_id} src={detection.source_camera}"
        cv2.putText(canvas, label, (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_preview(
    frame_a: Optional[FrameRecord],
    frame_b: Optional[FrameRecord],
    detections_a: dict[int, DetectionPose],
    detections_b: dict[int, DetectionPose],
    preview_width: int,
    status_lines: list[str],
) -> np.ndarray:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    image_a = frame_a.image if frame_a is not None else blank
    image_b = frame_b.image if frame_b is not None else blank
    panel_a = draw_detections(image_a, detections_a, "Camera A RGB")
    panel_b = draw_detections(image_b, detections_b, "Camera B RGB")

    width_each = max(1, preview_width // 2)
    panel_a = resize_to_width(panel_a, width_each)
    panel_b = resize_to_width(panel_b, width_each)
    target_height = min(panel_a.shape[0], panel_b.shape[0])
    panel_a = resize_to_height(panel_a, target_height)
    panel_b = resize_to_height(panel_b, target_height)
    preview = cv2.hconcat([panel_a, panel_b])

    y = 30
    for line in status_lines:
        cv2.putText(preview, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return preview


def prepare_output_paths(output_root: Path, session_name: str) -> tuple[Path, Path, Path]:
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = session_dir / "tag_pose_deltas.jsonl"
    summary_path = session_dir / "tracking_summary.json"
    return session_dir, jsonl_path, summary_path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_path = resolve_path(args.config)

    devices = list_realsense_devices()
    if args.list_devices:
        if not devices:
            print("No RealSense devices found.")
            return
        print("Detected RealSense devices:")
        for index, device in enumerate(devices, start=1):
            print(f"[{index}] name={device['name']}")
            print(f"    serial_no={device['serial_no']}")
            print(f"    firmware={device['firmware']}")
        return

    camera_a_cfg = config.get("camera_a", {})
    camera_b_cfg = config.get("camera_b", {})
    capture_cfg = config.get("capture", {})
    detector_cfg = config.get("apriltag", {})
    tracking_cfg = config.get("tracking", {})
    output_cfg = config.get("output", {})

    serial_a = str(camera_a_cfg.get("serial_no", "")).strip()
    serial_b = str(camera_b_cfg.get("serial_no", "")).strip()
    if not serial_a or not serial_b:
        raise ValueError("camera_a.serial_no and camera_b.serial_no must be set in the config file.")

    width = int(capture_cfg.get("width", 1280))
    height = int(capture_cfg.get("height", 720))
    fps = int(capture_cfg.get("fps", 30))
    preview_width = int(capture_cfg.get("preview_width", 1280))
    startup_timeout = float(capture_cfg.get("startup_timeout", 20.0))
    warmup_frames = int(capture_cfg.get("warmup_frames", 15))
    buffer_size = int(capture_cfg.get("buffer_size", 30))
    sync_max_delta_ms = float(capture_cfg.get("sync_max_delta_ms", 20.0))

    tracked_tag_ids = {int(tag_id) for tag_id in tracking_cfg.get("tag_ids", [])}
    if not tracked_tag_ids:
        raise ValueError("tracking.tag_ids must contain at least one tag id.")
    tag_size_m = float(tracking_cfg.get("tag_size_m", 0.0))
    if tag_size_m <= 0:
        raise ValueError("tracking.tag_size_m must be positive.")
    allowed_hamming = int(tracking_cfg.get("max_hamming", 0))

    output_root_value = str(output_cfg.get("output_root", str(DEFAULT_OUTPUT_ROOT)))
    output_root = resolve_path(output_root_value, PROJECT_ROOT)
    session_name = str(output_cfg.get("session_name", "")).strip()
    if not session_name:
        session_name = prompt_session_name()
    if not session_name:
        raise ValueError("session name is required.")
    session_dir, jsonl_path, summary_path = prepare_output_paths(output_root, session_name)

    intrinsics_cfg = config.get("intrinsics", {})
    camera_a_intrinsics_path, camera_matrix_a, _ = load_intrinsics(str(intrinsics_cfg.get("camera_a", "")), config_path.parent)
    camera_b_intrinsics_path, camera_matrix_b, _ = load_intrinsics(str(intrinsics_cfg.get("camera_b", "")), config_path.parent)

    detector = build_detector(detector_cfg)

    worker_a = CameraWorker(serial_a, "A", width, height, fps, buffer_size, warmup_frames)
    worker_b = CameraWorker(serial_b, "B", width, height, fps, buffer_size, warmup_frames)
    worker_a.start()
    worker_b.start()
    worker_a.wait_until_started(startup_timeout)
    worker_b.wait_until_started(startup_timeout)

    print(f"[INFO] Camera A serial: {serial_a}")
    print(f"[INFO] Camera B serial: {serial_b}")
    print(f"[INFO] Session directory: {session_dir}")
    print(f"[INFO] JSONL output: {jsonl_path}")
    print(f"[INFO] Intrinsics A: {camera_a_intrinsics_path}")
    print(f"[INFO] Intrinsics B: {camera_b_intrinsics_path}")
    print(f"[INFO] Tracking tag ids: {sorted(tracked_tag_ids)}")
    print("[INFO] Press Q in the preview window to quit.")

    previous_states: dict[int, Optional[tuple[str, np.ndarray]]] = {tag_id: None for tag_id in tracked_tag_ids}
    last_processed_frame_a: Optional[int] = None
    frames_written = 0
    valid_delta_counts: dict[int, int] = {tag_id: 0 for tag_id in tracked_tag_ids}
    window_name = "Dual D435i AprilTag Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_a, frame_b, sync_delta_ms = select_best_pair(worker_a.buffer, worker_b.buffer)
            if frame_a is None:
                blank = np.zeros((480, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera A frames...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, blank)
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    break
                continue

            if last_processed_frame_a == frame_a.frame_number:
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    break
                continue

            last_processed_frame_a = frame_a.frame_number
            detections_a = detect_tag_poses(detector, frame_a, camera_matrix_a, tag_size_m, tracked_tag_ids, allowed_hamming)
            detections_b: dict[int, DetectionPose] = {}
            if frame_b is not None and sync_delta_ms is not None and sync_delta_ms <= sync_max_delta_ms:
                detections_b = detect_tag_poses(detector, frame_b, camera_matrix_b, tag_size_m, tracked_tag_ids, allowed_hamming)

            record: dict[str, Any] = {
                "frame_a_number": frame_a.frame_number,
                "frame_a_host_timestamp_s": frame_a.host_timestamp_s,
                "frame_a_device_timestamp_ms": frame_a.device_timestamp_ms,
                "frame_b_number": frame_b.frame_number if frame_b is not None else None,
                "frame_b_host_timestamp_s": frame_b.host_timestamp_s if frame_b is not None else None,
                "frame_b_device_timestamp_ms": frame_b.device_timestamp_ms if frame_b is not None else None,
                "sync_delta_ms": sync_delta_ms,
                "tags": {},
            }

            status_lines = [
                f"frame_a={frame_a.frame_number}",
                f"sync_delta_ms={'N/A' if sync_delta_ms is None else f'{sync_delta_ms:.2f}'}",
            ]

            for tag_id in sorted(tracked_tag_ids):
                current_pose = choose_preferred_pose(tag_id, detections_a, detections_b)
                tag_entry: dict[str, Any] = {"visible": current_pose is not None}
                if current_pose is None:
                    previous_states[tag_id] = None
                    tag_entry["status"] = "missing"
                    record["tags"][str(tag_id)] = tag_entry
                    status_lines.append(f"tag {tag_id}: missing")
                    continue

                transform_camera_tag = current_pose.transform_camera_tag
                tag_entry["status"] = "tracked"
                tag_entry["source_camera"] = current_pose.source_camera
                tag_entry["transform_camera_tag"] = transform_camera_tag.tolist()
                tag_entry["translation_xyz"] = transform_camera_tag[:3, 3].tolist()
                tag_entry["quaternion_xyzw"] = rotation_matrix_to_quaternion(transform_camera_tag[:3, :3]).tolist()

                previous_state = previous_states[tag_id]
                if previous_state is not None and previous_state[0] == current_pose.source_camera:
                    _, previous_transform = previous_state
                    delta_transform = invert_transform(previous_transform) @ transform_camera_tag
                    delta_quaternion = rotation_matrix_to_quaternion(delta_transform[:3, :3])
                    tag_entry["delta_transform_prev_to_curr"] = delta_transform.tolist()
                    tag_entry["delta_translation_xyz"] = delta_transform[:3, 3].tolist()
                    tag_entry["delta_quaternion_xyzw"] = delta_quaternion.tolist()
                    tag_entry["delta_source_camera"] = current_pose.source_camera
                    valid_delta_counts[tag_id] += 1
                    status_lines.append(
                        f"tag {tag_id}: {current_pose.source_camera} d=({delta_transform[0,3]:.4f}, {delta_transform[1,3]:.4f}, {delta_transform[2,3]:.4f})"
                    )
                elif previous_state is not None:
                    previous_source, _ = previous_state
                    tag_entry["status"] = "source_switched"
                    tag_entry["previous_source_camera"] = previous_source
                    tag_entry["delta_transform_prev_to_curr"] = None
                    tag_entry["delta_translation_xyz"] = None
                    tag_entry["delta_quaternion_xyzw"] = None
                    tag_entry["delta_source_camera"] = None
                    status_lines.append(f"tag {tag_id}: source switch {previous_source}->{current_pose.source_camera}, delta skipped")
                else:
                    tag_entry["delta_transform_prev_to_curr"] = None
                    tag_entry["delta_translation_xyz"] = None
                    tag_entry["delta_quaternion_xyzw"] = None
                    tag_entry["delta_source_camera"] = None
                    status_lines.append(f"tag {tag_id}: {current_pose.source_camera} initialized")

                previous_states[tag_id] = (current_pose.source_camera, transform_camera_tag)
                record["tags"][str(tag_id)] = tag_entry

            append_jsonl(jsonl_path, record)
            frames_written += 1

            preview = build_preview(frame_a, frame_b, detections_a, detections_b, preview_width, status_lines[:8])
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                break
    finally:
        worker_a.stop()
        worker_b.stop()
        worker_a.join(timeout=5)
        worker_b.join(timeout=5)
        cv2.destroyAllWindows()

        summary = {
            "session_name": session_name,
            "output_root": str(output_root),
            "session_dir": str(session_dir),
            "jsonl_path": str(jsonl_path),
            "camera_a_serial_no": serial_a,
            "camera_b_serial_no": serial_b,
            "camera_a_intrinsics_file": str(camera_a_intrinsics_path),
            "camera_b_intrinsics_file": str(camera_b_intrinsics_path),
            "tag_ids": sorted(tracked_tag_ids),
            "tag_size_m": tag_size_m,
            "frames_written": frames_written,
            "tracking_mode": "prefer_camera_a_fallback_camera_b_skip_cross_camera_delta",
            "valid_delta_counts": {str(k): v for k, v in valid_delta_counts.items()},
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[DONE] Tracking summary written to: {summary_path}")


if __name__ == "__main__":
    main()
