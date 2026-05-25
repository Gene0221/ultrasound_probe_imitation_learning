from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyrealsense2 is required for this script. "
        "Install librealsense/pyrealsense2 first."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "capture.yaml"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"


@dataclass
class FrameRecord:
    image: np.ndarray
    device_timestamp_ms: float
    host_timestamp_s: float
    frame_number: int
    camera_label: str


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

                record = FrameRecord(
                    image=np.asanyarray(color_frame.get_data()).copy(),
                    device_timestamp_ms=float(color_frame.get_timestamp()),
                    host_timestamp_s=time.perf_counter(),
                    frame_number=int(color_frame.get_frame_number()),
                    camera_label=self.camera_label,
                )
                self.buffer.push(record)
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
    context = rs.context()
    devices = []
    for device in context.query_devices():
        devices.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial_no": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
        )
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture synchronized RGB image pairs from two D435i cameras."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config file.")
    parser.add_argument("--serial-a", default="", help="Serial number of camera A.")
    parser.add_argument("--serial-b", default="", help="Serial number of camera B.")
    parser.add_argument("--width", type=int, default=1280, help="RGB stream width.")
    parser.add_argument("--height", type=int, default=720, help="RGB stream height.")
    parser.add_argument("--fps", type=int, default=15, help="RGB stream FPS.")
    parser.add_argument("--buffer-size", type=int, default=10, help="Number of recent frames kept per camera.")
    parser.add_argument("--max-delta-ms", type=float, default=10.0, help="Maximum allowed host timestamp delta in milliseconds.")
    parser.add_argument("--session-name", default="", help="Session directory name under output-root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for captured pairs.")
    parser.add_argument("--preview-width", type=int, default=1280, help="Preview canvas width.")
    parser.add_argument("--startup-timeout", type=float, default=20.0, help="Camera startup timeout in seconds.")
    parser.add_argument("--warmup-frames", type=int, default=15, help="Frames to discard after pipeline start.")
    parser.add_argument("--list-devices", action="store_true", help="Only list detected RealSense devices and exit.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (SCRIPT_DIR / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return data


def apply_config_defaults(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    camera_a_cfg = config.get("camera_a", {})
    camera_b_cfg = config.get("camera_b", {})
    capture_cfg = config.get("capture", {})
    sync_cfg = config.get("synchronization", {})

    if not args.serial_a:
        args.serial_a = str(camera_a_cfg.get("serial_no", ""))
    if not args.serial_b:
        args.serial_b = str(camera_b_cfg.get("serial_no", ""))
    if args.width == 1280:
        args.width = int(capture_cfg.get("width", args.width))
    if args.height == 720:
        args.height = int(capture_cfg.get("height", args.height))
    if args.fps == 15:
        args.fps = int(capture_cfg.get("fps", args.fps))
    if args.buffer_size == 10:
        args.buffer_size = int(sync_cfg.get("buffer_size", args.buffer_size))
    if args.max_delta_ms == 10.0:
        args.max_delta_ms = float(sync_cfg.get("max_delta_ms", args.max_delta_ms))
    if args.session_name == "":
        args.session_name = str(capture_cfg.get("session_name", args.session_name))
    if args.output_root == str(DEFAULT_OUTPUT_ROOT):
        args.output_root = str(capture_cfg.get("output_root", args.output_root))
    if args.preview_width == 1280:
        args.preview_width = int(capture_cfg.get("preview_width", args.preview_width))
    if args.startup_timeout == 20.0:
        args.startup_timeout = float(capture_cfg.get("startup_timeout", args.startup_timeout))
    if args.warmup_frames == 15:
        args.warmup_frames = int(capture_cfg.get("warmup_frames", args.warmup_frames))
    return args


def resolve_serials(requested_a: str, requested_b: str) -> list[str]:
    devices = list_realsense_devices()
    if len(devices) < 2:
        raise RuntimeError(f"Expected at least 2 RealSense devices, found {len(devices)}.")

    if requested_a and requested_b:
        return [requested_a, requested_b]
    if requested_a and not requested_b:
        remaining = [d["serial_no"] for d in devices if d["serial_no"] != requested_a]
        if not remaining:
            raise RuntimeError("Could not auto-select camera B serial number from the currently connected devices.")
        return [requested_a, remaining[0]]
    if requested_b and not requested_a:
        remaining = [d["serial_no"] for d in devices if d["serial_no"] != requested_b]
        if not remaining:
            raise RuntimeError("Could not auto-select camera A serial number from the currently connected devices.")
        return [remaining[0], requested_b]
    return [devices[0]["serial_no"], devices[1]["serial_no"]]


def prompt_session_name(default_name: str) -> str:
    session_name = input(f"Enter session name [{default_name}]: ").strip()
    return session_name or default_name


def prepare_output_dirs(output_root: Path, session_name: str) -> tuple[Path, Path, Path, Path]:
    session_dir = output_root / session_name
    camera_a_dir = session_dir / "camera_a"
    camera_b_dir = session_dir / "camera_b"
    metadata_path = session_dir / "pairs_metadata.json"
    camera_a_dir.mkdir(parents=True, exist_ok=True)
    camera_b_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, camera_a_dir, camera_b_dir, metadata_path


def build_preview(
    frame_a: Optional[FrameRecord],
    frame_b: Optional[FrameRecord],
    preview_width: int,
    saved_items: int,
    last_delta_ms: Optional[float],
) -> np.ndarray:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    image_a = frame_a.image if frame_a is not None else blank
    panel_a = annotate_panel(image_a.copy(), "Camera A RGB")

    image_b = frame_b.image if frame_b is not None else blank
    panel_b = annotate_panel(image_b.copy(), "Camera B RGB")
    width_each = max(1, preview_width // 2)
    panel_a = resize_to_width(panel_a, width_each)
    panel_b = resize_to_width(panel_b, width_each)
    target_height = min(panel_a.shape[0], panel_b.shape[0])
    panel_a = resize_to_height(panel_a, target_height)
    panel_b = resize_to_height(panel_b, target_height)
    preview = cv2.hconcat([panel_a, panel_b])

    if last_delta_ms is None:
        delta_text = "delta_t: N/A"
    else:
        delta_text = f"delta_t: {last_delta_ms:.2f} ms"
    status = f"Saved items: {saved_items} | {delta_text} | Keys: S save, Q quit"
    cv2.putText(preview, status, (20, max(40, preview.shape[0] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def annotate_panel(image: np.ndarray, title: str) -> np.ndarray:
    cv2.putText(image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def select_frame_closest_to_time(frames: list[FrameRecord], target_host_time_s: float) -> Optional[FrameRecord]:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.host_timestamp_s - target_host_time_s))


def select_best_pair(buffer_a: FrameBuffer, buffer_b: FrameBuffer, trigger_time_s: float) -> tuple[Optional[FrameRecord], Optional[FrameRecord], Optional[float]]:
    frames_a = buffer_a.snapshot()
    frames_b = buffer_b.snapshot()
    if not frames_a or not frames_b:
        return None, None, None

    best_a = select_frame_closest_to_time(frames_a, trigger_time_s)
    if best_a is None:
        return None, None, None
    best_b = select_frame_closest_to_time(frames_b, best_a.host_timestamp_s)
    if best_b is None:
        return None, None, None

    delta_ms = abs(best_a.host_timestamp_s - best_b.host_timestamp_s) * 1000.0
    return best_a, best_b, delta_ms


def save_pair(
    pair_index: int,
    frame_a: FrameRecord,
    frame_b: FrameRecord,
    delta_ms: float,
    camera_a_dir: Path,
    camera_b_dir: Path,
    metadata_records: list[dict[str, object]],
) -> None:
    filename_a = f"camera_a_{pair_index:04d}.png"
    filename_b = f"camera_b_{pair_index:04d}.png"
    path_a = camera_a_dir / filename_a
    path_b = camera_b_dir / filename_b

    cv2.imwrite(str(path_a), frame_a.image)
    cv2.imwrite(str(path_b), frame_b.image)

    if delta_ms <= 5.0:
        match_status = "good"
    elif delta_ms <= 10.0:
        match_status = "acceptable"
    else:
        match_status = "reject"

    metadata_records.append(
        {
            "pair_index": pair_index,
            "camera_a_image": filename_a,
            "camera_b_image": filename_b,
            "camera_a_host_timestamp_s": frame_a.host_timestamp_s,
            "camera_b_host_timestamp_s": frame_b.host_timestamp_s,
            "camera_a_device_timestamp_ms": frame_a.device_timestamp_ms,
            "camera_b_device_timestamp_ms": frame_b.device_timestamp_ms,
            "camera_a_frame_number": frame_a.frame_number,
            "camera_b_frame_number": frame_b.frame_number,
            "timestamp_delta_ms": delta_ms,
            "match_status": match_status,
            "mode": "dual_camera",
        }
    )


def write_metadata(metadata_path: Path, payload: dict[str, object]) -> None:
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args = apply_config_defaults(args, config)

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

    serials = resolve_serials(args.serial_a, args.serial_b)
    print(f"[INFO] Camera A serial: {serials[0]}")
    print(f"[INFO] Camera B serial: {serials[1]}")

    default_session_name = time.strftime("rgb_capture_session_%Y%m%d_%H%M%S")
    session_name = args.session_name.strip() if args.session_name.strip() else prompt_session_name(default_session_name)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    else:
        output_root = output_root.resolve()
    session_dir, camera_a_dir, camera_b_dir, metadata_path = prepare_output_dirs(output_root, session_name)

    worker_a = CameraWorker(serials[0], "A", args.width, args.height, args.fps, args.buffer_size, args.warmup_frames)
    worker_a.start()
    worker_a.wait_until_started(args.startup_timeout)

    worker_b = CameraWorker(serials[1], "B", args.width, args.height, args.fps, args.buffer_size, args.warmup_frames)
    worker_b.start()
    worker_b.wait_until_started(args.startup_timeout)

    metadata_records: list[dict[str, object]] = []
    saved_items = 0
    last_delta_ms: Optional[float] = None

    print("[INFO] Cameras started. Press S to save, Q to quit.")
    window_name = "D435i RGB Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            latest_a = worker_a.buffer.latest()
            latest_b = worker_b.buffer.latest() if worker_b is not None else None
            preview = build_preview(
                latest_a,
                latest_b,
                args.preview_width,
                saved_items,
                last_delta_ms,
            )
            if latest_a is None or latest_b is None:
                cv2.putText(
                    preview,
                    "Waiting for camera frames...",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in {ord("q"), ord("Q")}:
                break

            if key in {ord("s"), ord("S")}:
                trigger_time_s = time.perf_counter()
                best_a, best_b, delta_ms = select_best_pair(worker_a.buffer, worker_b.buffer, trigger_time_s)
                if best_a is None or best_b is None or delta_ms is None:
                    print("[WARN] No valid synchronized pair is available yet.")
                    continue

                last_delta_ms = delta_ms
                if delta_ms > args.max_delta_ms:
                    print(f"[WARN] Pair rejected. delta_t = {delta_ms:.2f} ms exceeds threshold {args.max_delta_ms:.2f} ms.")
                    continue

                saved_items += 1
                save_pair(saved_items, best_a, best_b, delta_ms, camera_a_dir, camera_b_dir, metadata_records)
                print(
                    "[DONE] Saved pair "
                    f"{saved_items:04d} | "
                    f"A frame={best_a.frame_number}, B frame={best_b.frame_number}, "
                    f"delta_t={delta_ms:.2f} ms"
                )
    finally:
        worker_a.stop()
        worker_a.join(timeout=5)
        worker_b.stop()
        worker_b.join(timeout=5)
        cv2.destroyAllWindows()

        payload = {
            "session_name": session_name,
            "output_root": str(output_root),
            "session_dir": str(session_dir),
            "camera_a": {
                "serial_no": serials[0],
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
            },
            "camera_b": {
                "serial_no": serials[1],
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
            },
            "buffer_size": args.buffer_size,
            "saved_item_count": saved_items,
            "max_delta_ms": args.max_delta_ms,
            "items": metadata_records,
        }
        write_metadata(metadata_path, payload)
        print(f"[DONE] Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
