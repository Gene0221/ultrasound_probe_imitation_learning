from __future__ import annotations

import argparse
import json
import math
import threading
import time
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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "vision" / "apriltag_test.yaml"


@dataclass
class FrameRecord:
    image: np.ndarray
    device_timestamp_ms: float
    host_timestamp_s: float
    frame_number: int


@dataclass
class TagPose:
    tag_id: int
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    center_xy: tuple[float, float]
    corners_xy: np.ndarray
    decision_margin: float
    hamming: int


class CameraWorker(threading.Thread):
    def __init__(
        self,
        serial_no: str,
        width: int,
        height: int,
        fps: int,
        warmup_frames: int,
    ) -> None:
        super().__init__(daemon=True)
        self.serial_no = serial_no
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self._pipeline = rs.pipeline()
        self._stop_event = threading.Event()
        self._started_ok = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._latest: Optional[FrameRecord] = None

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

                record = FrameRecord(
                    image=np.asanyarray(color_frame.get_data()).copy(),
                    device_timestamp_ms=float(color_frame.get_timestamp()),
                    host_timestamp_s=time.time(),
                    frame_number=int(color_frame.get_frame_number()),
                )
                with self._lock:
                    self._latest = record
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
            raise TimeoutError(f"Camera did not start within {timeout_s:.1f} seconds.")
        if self._startup_error is not None:
            raise RuntimeError(f"Camera failed to start: {self._startup_error}") from self._startup_error

    def latest(self) -> Optional[FrameRecord]:
        with self._lock:
            return self._latest

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
        description="Single-camera AprilTag pose test with real-time visualization."
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
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


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
    image: np.ndarray,
    camera_matrix: np.ndarray,
    tag_size_m: float,
    allowed_tag_ids: Optional[set[int]],
    allowed_hamming: int,
) -> list[TagPose]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size_m)

    poses: list[TagPose] = []
    for detection in detections:
        tag_id = int(detection.tag_id)
        if allowed_tag_ids is not None and tag_id not in allowed_tag_ids:
            continue
        if int(detection.hamming) > allowed_hamming:
            continue
        poses.append(
            TagPose(
                tag_id=tag_id,
                rotation_matrix=np.array(detection.pose_R, dtype=np.float64),
                translation_vector=np.array(detection.pose_t, dtype=np.float64).reshape(3),
                center_xy=(float(detection.center[0]), float(detection.center[1])),
                corners_xy=np.array(detection.corners, dtype=np.float64),
                decision_margin=float(detection.decision_margin),
                hamming=int(detection.hamming),
            )
        )
    return poses


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_axes(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    axis_length_m: float,
) -> None:
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float32,
    )
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    projected, _ = cv2.projectPoints(axis_points, rvec, translation_vector.reshape(3, 1), camera_matrix, dist_coeffs)
    pts = projected.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(image, origin, tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(pts[3]), (255, 0, 0), 2)


def draw_pose_overlay(
    image: np.ndarray,
    poses: list[TagPose],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, "Single Camera AprilTag Pose Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    y = 80
    if not poses:
        cv2.putText(canvas, "No AprilTag detected", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
        return canvas

    for pose in poses:
        corners = pose.corners_xy.astype(int)
        cv2.polylines(canvas, [corners], True, (0, 255, 255), 2)
        center = (int(pose.center_xy[0]), int(pose.center_xy[1]))
        cv2.circle(canvas, center, 4, (0, 0, 255), -1)
        draw_axes(canvas, camera_matrix, dist_coeffs, pose.rotation_matrix, pose.translation_vector, axis_length_m)

        quaternion = rotation_matrix_to_quaternion(pose.rotation_matrix)
        translation = pose.translation_vector
        lines = [
            f"id={pose.tag_id} margin={pose.decision_margin:.1f} hamming={pose.hamming}",
            f"t = ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}) m",
            f"q = ({quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f})",
        ]
        for line in lines:
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            y += 26
        y += 10

    return canvas


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

    camera_cfg = config.get("camera", {})
    detector_cfg = config.get("apriltag", {})
    intrinsics_cfg = config.get("intrinsics", {})
    preview_cfg = config.get("preview", {})

    serial_no = str(camera_cfg.get("serial_no", "")).strip()
    if not serial_no:
        raise ValueError("camera.serial_no must be set in the config file.")

    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    fps = int(camera_cfg.get("fps", 30))
    warmup_frames = int(camera_cfg.get("warmup_frames", 15))
    startup_timeout = float(camera_cfg.get("startup_timeout", 20.0))

    intrinsics_path, camera_matrix, dist_coeffs = load_intrinsics(str(intrinsics_cfg.get("camera", "")), config_path.parent)

    tag_size_m = float(detector_cfg.get("tag_size_m", 0.0))
    if tag_size_m <= 0:
        raise ValueError("apriltag.tag_size_m must be positive.")
    tracked_ids_raw = detector_cfg.get("tag_ids", [])
    allowed_tag_ids = {int(tag_id) for tag_id in tracked_ids_raw} if tracked_ids_raw else None
    allowed_hamming = int(detector_cfg.get("max_hamming", 0))
    axis_length_m = float(detector_cfg.get("axis_length_m", tag_size_m * 0.5))

    detector = build_detector(detector_cfg)

    preview_width = int(preview_cfg.get("width", 1280))
    window_name = str(preview_cfg.get("window_name", "AprilTag Pose Test"))

    worker = CameraWorker(serial_no, width, height, fps, warmup_frames)
    worker.start()
    worker.wait_until_started(startup_timeout)

    print(f"[INFO] Camera serial: {serial_no}")
    print(f"[INFO] Intrinsics file: {intrinsics_path}")
    print(f"[INFO] Tag size: {tag_size_m:.4f} m")
    print(f"[INFO] Tracking tag ids: {'all' if allowed_tag_ids is None else sorted(allowed_tag_ids)}")
    print("[INFO] Press Q in the preview window to quit.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_frame_number: Optional[int] = None

    try:
        while True:
            frame = worker.latest()
            if frame is None:
                blank = np.zeros((480, preview_width, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera frames...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, blank)
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    break
                continue

            if last_frame_number == frame.frame_number:
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    break
                continue
            last_frame_number = frame.frame_number

            poses = detect_tag_poses(detector, frame.image, camera_matrix, tag_size_m, allowed_tag_ids, allowed_hamming)
            preview = draw_pose_overlay(frame.image, poses, camera_matrix, dist_coeffs, axis_length_m)
            preview = resize_to_width(preview, preview_width)
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                break
    finally:
        worker.stop()
        worker.join(timeout=5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
