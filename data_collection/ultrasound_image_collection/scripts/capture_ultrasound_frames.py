from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class VideoDeviceInfo:
    device: str
    name: str | None
    by_id: list[str]
    by_path: list[str]
    vendor_id: str | None
    product_id: str | None
    serial_number: str | None
    manufacturer: str | None
    product: str | None
    usb_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture timestamped ultrasound image frames from a USB video device on Ubuntu."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config file.")
    parser.add_argument("--list-devices", action="store_true", help="List detected /dev/video* devices and exit.")
    parser.add_argument(
        "--dry-run-resolve-device",
        action="store_true",
        help="Resolve the configured capture device and print the selected node without starting capture.",
    )
    parser.add_argument("--control-file", default=None, help="Optional JSON control file for start/pause/stop.")
    return parser.parse_args()


def resolve_workspace_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_workspace_path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected config object in {config_path}")
    return payload


def load_control_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"recording": True, "output_dir": None, "shutdown": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Control file must contain a JSON object: {path}")
    return payload


def read_text_if_exists(path: Path) -> str | None:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None
    return None


def find_usb_metadata(device_sysfs_path: Path) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {
        "vendor_id": None,
        "product_id": None,
        "serial_number": None,
        "manufacturer": None,
        "product": None,
        "usb_path": None,
    }
    current = device_sysfs_path.resolve()
    for candidate in [current, *current.parents]:
        if metadata["vendor_id"] is None:
            metadata["vendor_id"] = read_text_if_exists(candidate / "idVendor")
        if metadata["product_id"] is None:
            metadata["product_id"] = read_text_if_exists(candidate / "idProduct")
        if metadata["serial_number"] is None:
            metadata["serial_number"] = read_text_if_exists(candidate / "serial")
        if metadata["manufacturer"] is None:
            metadata["manufacturer"] = read_text_if_exists(candidate / "manufacturer")
        if metadata["product"] is None:
            metadata["product"] = read_text_if_exists(candidate / "product")
        if metadata["usb_path"] is None:
            devpath = read_text_if_exists(candidate / "devpath")
            busnum = read_text_if_exists(candidate / "busnum")
            if devpath and busnum:
                metadata["usb_path"] = f"{busnum}-{devpath}"
    return metadata


def discover_video_devices() -> list[VideoDeviceInfo]:
    sysfs_root = Path("/sys/class/video4linux")
    by_id_root = Path("/dev/v4l/by-id")
    by_path_root = Path("/dev/v4l/by-path")
    if not sysfs_root.exists():
        return []

    by_id_map: dict[Path, list[str]] = {}
    if by_id_root.exists():
        for entry in sorted(by_id_root.iterdir()):
            try:
                target = entry.resolve()
            except OSError:
                continue
            by_id_map.setdefault(target, []).append(str(entry))

    by_path_map: dict[Path, list[str]] = {}
    if by_path_root.exists():
        for entry in sorted(by_path_root.iterdir()):
            try:
                target = entry.resolve()
            except OSError:
                continue
            by_path_map.setdefault(target, []).append(str(entry))

    devices: list[VideoDeviceInfo] = []
    for sysfs_node in sorted(sysfs_root.glob("video*")):
        device_path = Path("/dev") / sysfs_node.name
        metadata = find_usb_metadata(sysfs_node / "device")
        devices.append(
            VideoDeviceInfo(
                device=str(device_path),
                name=read_text_if_exists(sysfs_node / "name"),
                by_id=by_id_map.get(device_path.resolve(), []),
                by_path=by_path_map.get(device_path.resolve(), []),
                vendor_id=metadata["vendor_id"],
                product_id=metadata["product_id"],
                serial_number=metadata["serial_number"],
                manufacturer=metadata["manufacturer"],
                product=metadata["product"],
                usb_path=metadata["usb_path"],
            )
        )
    return devices


def format_device_info(device: VideoDeviceInfo) -> str:
    return (
        f"{device.device}: "
        f"name={device.name!r}, vendor_id={device.vendor_id!r}, product_id={device.product_id!r}, "
        f"serial={device.serial_number!r}, manufacturer={device.manufacturer!r}, "
        f"product={device.product!r}, usb_path={device.usb_path!r}, "
        f"by_id={device.by_id}, by_path={device.by_path}"
    )


def print_devices() -> None:
    devices = discover_video_devices()
    if not devices:
        print("no /dev/video* devices found")
        return
    print("video devices:")
    for device in devices:
        print(f"  {format_device_info(device)}")


def normalize(value: Any) -> str:
    return "" if value is None else str(value).strip()


def is_probably_capture_node(device: VideoDeviceInfo) -> bool:
    lowered_name = (device.name or "").lower()
    if "metadata" in lowered_name:
        return False
    if any(link.endswith("index0") for link in device.by_id):
        return True
    if any(link.endswith("index1") for link in device.by_id) and not any(link.endswith("index0") for link in device.by_id):
        return False
    return True


def resolve_device(config: dict[str, Any]) -> VideoDeviceInfo:
    device_cfg = config.get("device", {})
    requested_path = normalize(device_cfg.get("path", "AUTO"))
    requested_by_id = normalize(device_cfg.get("by_id", ""))
    requested_by_path = normalize(device_cfg.get("by_path", ""))
    requested_name = normalize(device_cfg.get("name_contains", "")).lower()
    requested_serial = normalize(device_cfg.get("serial_number", "")).lower()
    requested_vendor = normalize(device_cfg.get("vendor_id", "")).lower()
    requested_product = normalize(device_cfg.get("product_id", "")).lower()
    requested_usb_path = normalize(device_cfg.get("usb_path_contains", "")).lower()

    devices = discover_video_devices()
    if requested_path and requested_path.upper() != "AUTO":
        direct_path = Path(requested_path)
        if direct_path.exists():
            resolved_direct = str(direct_path.resolve())
            for device in devices:
                if device.device == resolved_direct:
                    return device
            return VideoDeviceInfo(
                device=resolved_direct,
                name=None,
                by_id=[],
                by_path=[],
                vendor_id=None,
                product_id=None,
                serial_number=None,
                manufacturer=None,
                product=None,
                usb_path=None,
            )
        raise RuntimeError(f"Configured video device path does not exist: {requested_path}")

    candidates = [device for device in devices if is_probably_capture_node(device)]

    if requested_by_id:
        candidates = [
            device for device in candidates if any(requested_by_id in link.lower() for link in device.by_id)
        ]
    if requested_by_path:
        candidates = [
            device for device in candidates if any(requested_by_path in link.lower() for link in device.by_path)
        ]
    if requested_name:
        candidates = [device for device in candidates if requested_name in normalize(device.name).lower()]
    if requested_serial:
        candidates = [
            device for device in candidates if requested_serial == normalize(device.serial_number).lower()
        ]
    if requested_vendor:
        candidates = [
            device for device in candidates if requested_vendor == normalize(device.vendor_id).lower()
        ]
    if requested_product:
        candidates = [
            device for device in candidates if requested_product == normalize(device.product_id).lower()
        ]
    if requested_usb_path:
        candidates = [
            device for device in candidates if requested_usb_path in normalize(device.usb_path).lower()
        ]

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "No video device matched the configured filters. "
            f"Available devices: {[format_device_info(device) for device in devices]}"
        )
    raise RuntimeError(
        "Multiple video devices matched the configured filters. "
        "Refine device.path, device.by_id, device.by_path, device.serial_number, or other filters. "
        f"Matches: {[format_device_info(device) for device in candidates]}"
    )


def open_capture(device_path: str, capture_cfg: dict[str, Any]) -> cv2.VideoCapture:
    backend_name = normalize(capture_cfg.get("backend", "V4L2")).upper()
    if backend_name == "V4L2":
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    capture = cv2.VideoCapture(device_path, backend)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video capture device: {device_path}")

    width = capture_cfg.get("width")
    height = capture_cfg.get("height")
    fps = capture_cfg.get("fps")
    fourcc = normalize(capture_cfg.get("fourcc", ""))

    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps is not None:
        capture.set(cv2.CAP_PROP_FPS, float(fps))
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))

    return capture


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def finalize_session(
    summary_path: Path | None,
    device_info: VideoDeviceInfo,
    session_id: str | None,
    started_at_s: float | None,
    finished_at_s: float | None,
    frames_saved: int,
    image_format: str,
    target_hz: float,
    capture: cv2.VideoCapture,
) -> None:
    if summary_path is None or session_id is None:
        return
    duration_s = None
    achieved_hz = None
    if started_at_s is not None and finished_at_s is not None and finished_at_s >= started_at_s:
        duration_s = finished_at_s - started_at_s
        if duration_s > 0:
            achieved_hz = frames_saved / duration_s
    write_json(
        summary_path,
        {
            "session_id": session_id,
            "device_path": device_info.device,
            "device_name": device_info.name,
            "device_by_id": device_info.by_id,
            "device_by_path": device_info.by_path,
            "vendor_id": device_info.vendor_id,
            "product_id": device_info.product_id,
            "serial_number": device_info.serial_number,
            "usb_path": device_info.usb_path,
            "image_format": image_format,
            "target_hz": target_hz,
            "frames_saved": frames_saved,
            "started_at_s": started_at_s,
            "finished_at_s": finished_at_s,
            "duration_s": duration_s,
            "achieved_hz": achieved_hz,
            "reported_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "reported_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "reported_fps": capture.get(cv2.CAP_PROP_FPS),
            "reported_fourcc": int(capture.get(cv2.CAP_PROP_FOURCC)),
        },
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.list_devices:
        print_devices()
        return

    device_info = resolve_device(config)
    if args.dry_run_resolve_device:
        print(device_info.device)
        print(format_device_info(device_info))
        return

    output_cfg = config.get("output", {})
    capture_cfg = config.get("capture", {})
    recording_cfg = config.get("recording", {})
    output_root = resolve_workspace_path(output_cfg.get("output_root", "output"))
    output_root.mkdir(parents=True, exist_ok=True)

    image_format = normalize(output_cfg.get("image_format", "png")).lower() or "png"
    if image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("output.image_format must be one of: png, jpg, jpeg")
    image_ext = ".jpg" if image_format in {"jpg", "jpeg"} else ".png"
    target_hz = float(recording_cfg.get("target_hz", capture_cfg.get("fps", 30.0)))
    save_period_s = 1.0 / max(target_hz, 1e-6)
    timestamps_file_name = str(output_cfg.get("timestamps_file_name", "timestamps.jsonl"))
    images_subdir = str(output_cfg.get("images_subdir", "images"))
    frame_prefix = str(output_cfg.get("frame_prefix", "frame_"))
    jpg_quality = int(output_cfg.get("jpg_quality", 95))
    png_compression = int(output_cfg.get("png_compression", 3))
    print_stdout = bool(output_cfg.get("print_stdout", True))
    control_file = Path(args.control_file).resolve() if args.control_file else None

    capture = open_capture(device_info.device, capture_cfg)
    stop_requested = False
    timestamps_handle: Any | None = None
    active_output_dir: Path | None = None
    active_images_dir: Path | None = None
    active_summary_path: Path | None = None
    session_frame_index = 0
    last_saved_at_monotonic_s: float | None = None
    session_started_at_s: float | None = None
    session_id: str | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[INFO] Ultrasound logger received signal {signum}; stopping...", file=sys.stderr)

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        print(f"[INFO] Capturing from {device_info.device}")
        print(f"[INFO] Device info: {format_device_info(device_info)}")
        print(
            "[INFO] Reported capture properties: "
            f"width={capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}, "
            f"height={capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, "
            f"fps={capture.get(cv2.CAP_PROP_FPS):.3f}"
        )

        while not stop_requested:
            control_state = load_control_state(control_file)
            if bool(control_state.get("shutdown", False)):
                stop_requested = True
                break

            recording = bool(control_state.get("recording", control_file is None))
            output_dir_value = control_state.get("output_dir")
            requested_output_dir = Path(output_dir_value).resolve() if output_dir_value else output_root

            if recording and active_output_dir != requested_output_dir:
                if timestamps_handle is not None:
                    timestamps_handle.close()
                    finalize_session(
                        summary_path=active_summary_path,
                        device_info=device_info,
                        session_id=session_id,
                        started_at_s=session_started_at_s,
                        finished_at_s=time.time(),
                        frames_saved=session_frame_index,
                        image_format=image_format,
                        target_hz=target_hz,
                        capture=capture,
                    )
                requested_output_dir.mkdir(parents=True, exist_ok=True)
                active_images_dir = requested_output_dir / images_subdir
                active_images_dir.mkdir(parents=True, exist_ok=True)
                timestamps_path = requested_output_dir / timestamps_file_name
                active_summary_path = requested_output_dir / str(output_cfg.get("summary_file_name", "summary.json"))
                timestamps_handle = timestamps_path.open("w", encoding="utf-8")
                active_output_dir = requested_output_dir
                session_frame_index = 0
                last_saved_at_monotonic_s = None
                session_started_at_s = time.time()
                session_id = requested_output_dir.name
            elif not recording and active_output_dir is not None:
                if timestamps_handle is not None:
                    timestamps_handle.close()
                    timestamps_handle = None
                finalize_session(
                    summary_path=active_summary_path,
                    device_info=device_info,
                    session_id=session_id,
                    started_at_s=session_started_at_s,
                    finished_at_s=time.time(),
                    frames_saved=session_frame_index,
                    image_format=image_format,
                    target_hz=target_hz,
                    capture=capture,
                )
                active_output_dir = None
                active_images_dir = None
                active_summary_path = None
                session_frame_index = 0
                last_saved_at_monotonic_s = None
                session_started_at_s = None
                session_id = None

            ok, frame = capture.read()
            host_timestamp_s = time.time()
            now_monotonic_s = time.monotonic()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if not recording or timestamps_handle is None or active_images_dir is None:
                continue

            if last_saved_at_monotonic_s is not None and now_monotonic_s - last_saved_at_monotonic_s < save_period_s:
                continue

            image_name = f"{frame_prefix}{session_frame_index:06d}{image_ext}"
            image_path = active_images_dir / image_name
            if image_ext == ".png":
                success = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
            else:
                success = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            if not success:
                raise RuntimeError(f"Failed to write image: {image_path}")

            payload: dict[str, Any] = {
                "image": f"{images_subdir}/{image_name}",
                "host_timestamp_s": host_timestamp_s,
                "frame_index": session_frame_index,
                "device_path": device_info.device,
                "shape_hw": [int(frame.shape[0]), int(frame.shape[1])],
            }
            if bool(recording_cfg.get("include_device_name", True)):
                payload["device_name"] = device_info.name
            timestamps_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            timestamps_handle.flush()

            session_frame_index += 1
            last_saved_at_monotonic_s = now_monotonic_s
            if print_stdout:
                print(
                    f"saved frame={session_frame_index:06d} "
                    f"ts={host_timestamp_s:.6f} path={image_path.name}"
                )
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted ultrasound acquisition. Exiting cleanly.")
    finally:
        if timestamps_handle is not None:
            timestamps_handle.close()
            finalize_session(
                summary_path=active_summary_path,
                device_info=device_info,
                session_id=session_id,
                started_at_s=session_started_at_s,
                finished_at_s=time.time(),
                frames_saved=session_frame_index,
                image_format=image_format,
                target_hz=target_hz,
                capture=capture,
            )
        capture.release()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
