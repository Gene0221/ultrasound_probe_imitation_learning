from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (
    build_capture_filename,
    count_complete_frame_sets,
    detect_corners_in_image,
    ensure_dir,
    get_next_frame_index,
    get_sensor_ids,
    get_sensor_paths,
    load_config,
    prepare_session_storage,
    prompt_session_name,
)

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - import is environment dependent
    rs = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one D435i calibration session interactively.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config.")
    parser.add_argument("--session-name", default="", help="Optional save name for this calibration session.")
    parser.add_argument("--target-count", type=int, default=0, help="Override target number of synchronized captures.")
    parser.add_argument(
        "--skip-auto-calibration",
        action="store_true",
        help="Only capture data and do not start calibration automatically.",
    )
    return parser.parse_args()


def require_realsense() -> None:
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is not installed. Install it first, for example with: pip install pyrealsense2"
        )


def configure_pipeline(config: dict) -> tuple[object, object]:
    require_realsense()
    capture_cfg = config.get("capture", {})
    streams_cfg = capture_cfg.get("streams", {})
    color_cfg = streams_cfg.get("color", {})
    infrared_cfg = streams_cfg.get("infrared", {})

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.color,
        int(color_cfg.get("width", 1920)),
        int(color_cfg.get("height", 1080)),
        rs.format.bgr8,
        int(color_cfg.get("fps", 30)),
    )
    rs_config.enable_stream(
        rs.stream.infrared,
        1,
        int(infrared_cfg.get("width", 1280)),
        int(infrared_cfg.get("height", 720)),
        rs.format.y8,
        int(infrared_cfg.get("fps", 30)),
    )
    rs_config.enable_stream(
        rs.stream.infrared,
        2,
        int(infrared_cfg.get("width", 1280)),
        int(infrared_cfg.get("height", 720)),
        rs.format.y8,
        int(infrared_cfg.get("fps", 30)),
    )
    profile = pipeline.start(rs_config)
    return pipeline, profile


def warm_up_pipeline(pipeline: object, frame_count: int) -> None:
    for _ in range(frame_count):
        pipeline.wait_for_frames()


def read_frame_triplet(pipeline: object) -> dict[str, np.ndarray]:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    ir_left_frame = frames.get_infrared_frame(1)
    ir_right_frame = frames.get_infrared_frame(2)
    if not color_frame or not ir_left_frame or not ir_right_frame:
        raise RuntimeError("Failed to receive RGB + infrared frame triplet from D435i.")

    return {
        "rgb": np.asanyarray(color_frame.get_data()),
        "ir_left": np.asanyarray(ir_left_frame.get_data()),
        "ir_right": np.asanyarray(ir_right_frame.get_data()),
    }


def build_preview_image(images: dict[str, np.ndarray], preview_width: int) -> np.ndarray:
    panels = [
        annotate_panel(images["rgb"], "RGB"),
        annotate_panel(images["ir_left"], "IR Left"),
        annotate_panel(images["ir_right"], "IR Right"),
    ]
    panel_width = max(1, preview_width // len(panels))
    resized_panels = [resize_to_width(panel, panel_width) for panel in panels]
    target_height = min(panel.shape[0] for panel in resized_panels)
    aligned_panels = [resize_to_height(panel, target_height) for panel in resized_panels]
    return cv2.hconcat(aligned_panels)


def annotate_panel(image: np.ndarray, title: str) -> np.ndarray:
    if image.ndim == 2:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        panel = image.copy()
    cv2.putText(panel, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return panel


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_frame_triplet(config: dict, session_name: str, frame_index: int, images: dict[str, np.ndarray]) -> None:
    image_extension = config.get("capture", {}).get("image_extension", ".png")
    for sensor_id, image in images.items():
        sensor_paths = get_sensor_paths(config, session_name, sensor_id)
        ensure_dir(sensor_paths.image_dir)
        output_path = sensor_paths.image_dir / build_capture_filename(sensor_id, frame_index, image_extension)
        cv2.imwrite(str(output_path), image)


def print_detection_summary(config: dict, images: dict[str, np.ndarray]) -> None:
    parts: list[str] = []
    for sensor_id, image in images.items():
        success, _ = detect_corners_in_image(image, config)
        parts.append(f"{sensor_id}={'OK' if success else 'MISS'}")
    print("[INFO] Corner check: " + ", ".join(parts))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    capture_cfg = config.get("capture", {})
    target_count = args.target_count or int(capture_cfg.get("target_frame_count", 30))
    default_session_name = capture_cfg.get("default_session_name", "d435i_session")
    session_name = args.session_name.strip() if args.session_name.strip() else prompt_session_name(default_session_name)
    strategy = prepare_session_storage(config, session_name)

    required_sensors = get_sensor_ids(config)
    current_count = count_complete_frame_sets(config, session_name, required_sensors)
    next_frame_index = get_next_frame_index(config, session_name, required_sensors)

    print(f"[INFO] Session name: {session_name}")
    print(f"[INFO] Existing strategy: {strategy}")
    print(f"[INFO] Already saved synchronized frame sets: {current_count}")
    print(f"[INFO] Target synchronized frame sets: {target_count}")

    pipeline, _profile = configure_pipeline(config)
    warm_up_pipeline(pipeline, int(capture_cfg.get("warmup_frames", 15)))
    window_name = capture_cfg.get("preview_window_name", "D435i Calibration Preview")
    preview_width = int(capture_cfg.get("preview_width", 1800))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while current_count < target_count:
            images = read_frame_triplet(pipeline)
            preview_image = build_preview_image(images, preview_width)
            status_line = (
                f"Session: {session_name} | Saved: {current_count}/{target_count} | "
                "Keys: C capture, Q quit"
            )
            cv2.putText(
                preview_image,
                status_line,
                (20, max(60, preview_image.shape[0] - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview_image)
            key = cv2.waitKey(1) & 0xFF

            if key in {ord("q"), ord("Q")}:
                print("[INFO] Capture stopped by user.")
                break

            if key in {ord("c"), ord("C"), 13, 32}:
                print_detection_summary(config, images)
                save_frame_triplet(config, session_name, next_frame_index, images)
                print(f"[DONE] Saved synchronized frame set {next_frame_index:04d}")
                current_count += 1
                next_frame_index += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if current_count < target_count:
        print(f"[INFO] Capture finished early with {current_count} synchronized frame sets.")
        return

    print(f"[DONE] Reached target of {target_count} synchronized frame sets.")
    if args.skip_auto_calibration:
        print("[INFO] Auto calibration skipped.")
        return

    run_script = Path(__file__).resolve().parent / "run_single_device_calibration.py"
    subprocess.run(
        [sys.executable, str(run_script), "--config", args.config, "--session-name", session_name],
        check=True,
    )


if __name__ == "__main__":
    main()
