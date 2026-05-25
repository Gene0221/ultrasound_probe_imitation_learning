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
        "pupil_apriltags is required for AprilTag testing. Install it before running this script."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "vision" / "apriltag_delta_consistency.yaml"


@dataclass
class FrameRecord:
    image: np.ndarray
    device_timestamp_ms: float
    host_timestamp_s: float
    frame_number: int
    camera_label: str


@dataclass
class TagPose:
    tag_id: int
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    center_xy: tuple[float, float]
    corners_xy: np.ndarray
    decision_margin: float
    hamming: int


@dataclass
class DeltaRecord:
    start_frame_number: int
    end_frame_number: int
    start_host_timestamp_s: float
    end_host_timestamp_s: float
    transform_delta: np.ndarray


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
                        host_timestamp_s=time.time(),
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
        description="Compare adjacent-frame AprilTag pose deltas computed independently by camera A and camera B."
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


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    rotation_err = rotation_a.T @ rotation_b
    cos_theta = (np.trace(rotation_err) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


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


def detect_tag_pose(
    detector: Detector,
    image: np.ndarray,
    camera_matrix: np.ndarray,
    tag_size_m: float,
    tag_id: int,
    allowed_hamming: int,
) -> Optional[TagPose]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size_m)

    for detection in detections:
        if int(detection.tag_id) != tag_id:
            continue
        if int(detection.hamming) > allowed_hamming:
            continue
        return TagPose(
            tag_id=int(detection.tag_id),
            rotation_matrix=np.array(detection.pose_R, dtype=np.float64),
            translation_vector=np.array(detection.pose_t, dtype=np.float64).reshape(3),
            center_xy=(float(detection.center[0]), float(detection.center[1])),
            corners_xy=np.array(detection.corners, dtype=np.float64),
            decision_margin=float(detection.decision_margin),
            hamming=int(detection.hamming),
        )
    return None


def pose_to_transform(pose: TagPose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose.rotation_matrix
    transform[:3, 3] = pose.translation_vector
    return transform


def draw_pose_box(
    image: np.ndarray,
    pose: Optional[TagPose],
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    if pose is None:
        cv2.putText(canvas, "Tag missing", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
        return canvas
    corners = pose.corners_xy.astype(int)
    cv2.polylines(canvas, [corners], True, color, 2)
    center = (int(pose.center_xy[0]), int(pose.center_xy[1]))
    cv2.circle(canvas, center, 4, (0, 0, 255), -1)
    translation = pose.translation_vector
    quaternion = rotation_matrix_to_quaternion(pose.rotation_matrix)
    lines = [
        f"id={pose.tag_id} margin={pose.decision_margin:.1f}",
        f"t=({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}) m",
        f"q=({quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f})",
    ]
    y = 80
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += 24
    return canvas


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def build_preview(
    frame_a: Optional[FrameRecord],
    frame_b: Optional[FrameRecord],
    pose_a: Optional[TagPose],
    pose_b: Optional[TagPose],
    preview_width: int,
    status_lines: list[str],
) -> np.ndarray:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    image_a = frame_a.image if frame_a is not None else blank
    image_b = frame_b.image if frame_b is not None else blank
    panel_a = draw_pose_box(image_a, pose_a, "Camera A", (0, 255, 255))
    panel_b = draw_pose_box(image_b, pose_b, "Camera B", (255, 255, 0))

    width_each = max(1, preview_width // 2)
    panel_a = resize_to_width(panel_a, width_each)
    panel_b = resize_to_width(panel_b, width_each)
    target_height = min(panel_a.shape[0], panel_b.shape[0])
    panel_a = resize_to_height(panel_a, target_height)
    panel_b = resize_to_height(panel_b, target_height)
    preview = cv2.hconcat([panel_a, panel_b])

    y = 30
    for line in status_lines:
        cv2.putText(preview, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26
    return preview


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
    compare_cfg = config.get("compare", {})
    preview_cfg = config.get("preview", {})
    intrinsics_cfg = config.get("intrinsics", {})

    serial_a = str(camera_a_cfg.get("serial_no", "")).strip()
    serial_b = str(camera_b_cfg.get("serial_no", "")).strip()
    if not serial_a or not serial_b:
        raise ValueError("camera_a.serial_no and camera_b.serial_no must be set in the config file.")

    width = int(capture_cfg.get("width", 1280))
    height = int(capture_cfg.get("height", 720))
    fps = int(capture_cfg.get("fps", 30))
    preview_width = int(preview_cfg.get("width", 1280))
    warmup_frames = int(capture_cfg.get("warmup_frames", 15))
    startup_timeout = float(capture_cfg.get("startup_timeout", 20.0))
    buffer_size = int(capture_cfg.get("buffer_size", 30))
    sync_max_delta_ms = float(capture_cfg.get("sync_max_delta_ms", 20.0))
    print_interval_s = float(compare_cfg.get("print_interval_seconds", 3.0))

    tag_id = int(compare_cfg.get("tag_id", 0))
    tag_size_m = float(detector_cfg.get("tag_size_m", 0.0))
    if tag_size_m <= 0:
        raise ValueError("apriltag.tag_size_m must be positive.")
    allowed_hamming = int(detector_cfg.get("max_hamming", 0))

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
    print(f"[INFO] Intrinsics A: {camera_a_intrinsics_path}")
    print(f"[INFO] Intrinsics B: {camera_b_intrinsics_path}")
    print(f"[INFO] Comparing tag id: {tag_id}")
    print(f"[INFO] Print interval: {print_interval_s:.1f} s")
    print("[INFO] Press Q in the preview window to quit.")

    previous_pose_a: Optional[np.ndarray] = None
    previous_pose_b: Optional[np.ndarray] = None
    previous_frame_a: Optional[FrameRecord] = None
    previous_frame_b: Optional[FrameRecord] = None
    last_processed_frame_a: Optional[int] = None
    window_name = str(preview_cfg.get("window_name", "AprilTag Delta Consistency Test"))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_report_time = time.perf_counter()
    interval_translation_errors: list[float] = []
    interval_rotation_errors_deg: list[float] = []
    interval_pair_count = 0

    try:
        while True:
            frame_a, frame_b, sync_delta_ms = select_best_pair(worker_a.buffer, worker_b.buffer)
            if frame_a is None:
                blank = np.zeros((480, preview_width, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera A frames...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, blank)
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    break
                continue

            if last_processed_frame_a == frame_a.frame_number:
                now = time.perf_counter()
                if now - last_report_time >= print_interval_s:
                    if interval_pair_count > 0:
                        print(
                            "[REPORT] "
                            f"count={interval_pair_count} | "
                            f"translation_error_mean={np.mean(interval_translation_errors):.6f} m | "
                            f"translation_error_max={np.max(interval_translation_errors):.6f} m | "
                            f"rotation_error_mean={np.mean(interval_rotation_errors_deg):.6f} deg | "
                            f"rotation_error_max={np.max(interval_rotation_errors_deg):.6f} deg"
                        )
                    else:
                        print("[REPORT] No valid A/B delta pairs in the last interval.")
                    interval_translation_errors.clear()
                    interval_rotation_errors_deg.clear()
                    interval_pair_count = 0
                    last_report_time = now
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    break
                continue

            last_processed_frame_a = frame_a.frame_number
            pose_a = detect_tag_pose(detector, frame_a.image, camera_matrix_a, tag_size_m, tag_id, allowed_hamming)
            pose_b = None
            if frame_b is not None and sync_delta_ms is not None and sync_delta_ms <= sync_max_delta_ms:
                pose_b = detect_tag_pose(detector, frame_b.image, camera_matrix_b, tag_size_m, tag_id, allowed_hamming)

            status_lines = [
                f"frame_a={frame_a.frame_number}",
                f"frame_b={frame_b.frame_number if frame_b is not None else 'N/A'}",
                f"sync_delta_ms={'N/A' if sync_delta_ms is None else f'{sync_delta_ms:.2f}'}",
            ]

            current_transform_a = pose_to_transform(pose_a) if pose_a is not None else None
            current_transform_b = pose_to_transform(pose_b) if pose_b is not None else None
            current_delta_a: Optional[DeltaRecord] = None
            current_delta_b: Optional[DeltaRecord] = None

            if pose_a is not None and previous_pose_a is not None and previous_frame_a is not None:
                current_delta_a = DeltaRecord(
                    start_frame_number=previous_frame_a.frame_number,
                    end_frame_number=frame_a.frame_number,
                    start_host_timestamp_s=previous_frame_a.host_timestamp_s,
                    end_host_timestamp_s=frame_a.host_timestamp_s,
                    transform_delta=invert_transform(previous_pose_a) @ current_transform_a,
                )
            if pose_b is not None and previous_pose_b is not None and previous_frame_b is not None and frame_b is not None:
                current_delta_b = DeltaRecord(
                    start_frame_number=previous_frame_b.frame_number,
                    end_frame_number=frame_b.frame_number,
                    start_host_timestamp_s=previous_frame_b.host_timestamp_s,
                    end_host_timestamp_s=frame_b.host_timestamp_s,
                    transform_delta=invert_transform(previous_pose_b) @ current_transform_b,
                )

            if current_delta_a is not None and current_delta_b is not None:
                delta_translation_a = current_delta_a.transform_delta[:3, 3]
                delta_translation_b = current_delta_b.transform_delta[:3, 3]
                translation_error = float(np.linalg.norm(delta_translation_a - delta_translation_b))
                rotation_error = rotation_error_deg(
                    current_delta_a.transform_delta[:3, :3],
                    current_delta_b.transform_delta[:3, :3],
                )
                interval_translation_errors.append(translation_error)
                interval_rotation_errors_deg.append(rotation_error)
                interval_pair_count += 1
                status_lines.append(f"translation_error={translation_error:.6f} m")
                status_lines.append(f"rotation_error={rotation_error:.6f} deg")
            else:
                status_lines.append("delta comparison waiting...")

            if pose_a is None:
                previous_pose_a = None
                previous_frame_a = None
            else:
                previous_pose_a = current_transform_a
                previous_frame_a = frame_a

            if pose_b is None or frame_b is None:
                previous_pose_b = None
                previous_frame_b = None
            else:
                previous_pose_b = current_transform_b
                previous_frame_b = frame_b

            preview = build_preview(frame_a, frame_b, pose_a, pose_b, preview_width, status_lines[:8])
            cv2.imshow(window_name, preview)

            now = time.perf_counter()
            if now - last_report_time >= print_interval_s:
                if interval_pair_count > 0:
                    print(
                        "[REPORT] "
                        f"count={interval_pair_count} | "
                        f"translation_error_mean={np.mean(interval_translation_errors):.6f} m | "
                        f"translation_error_max={np.max(interval_translation_errors):.6f} m | "
                        f"rotation_error_mean={np.mean(interval_rotation_errors_deg):.6f} deg | "
                        f"rotation_error_max={np.max(interval_rotation_errors_deg):.6f} deg"
                    )
                else:
                    print("[REPORT] No valid A/B delta pairs in the last interval.")
                interval_translation_errors.clear()
                interval_rotation_errors_deg.clear()
                interval_pair_count = 0
                last_report_time = now

            if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                break
    finally:
        worker_a.stop()
        worker_b.stop()
        worker_a.join(timeout=5)
        worker_b.join(timeout=5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
