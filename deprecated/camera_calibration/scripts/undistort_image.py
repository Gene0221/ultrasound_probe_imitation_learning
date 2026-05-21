from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import ensure_dir, get_sensor_paths, load_config, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Undistort one image or a directory of images.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config.")
    parser.add_argument("--session-name", required=True, help="Saved capture session name.")
    parser.add_argument("--sensor-id", required=True, help="Sensor ID defined in config.")
    parser.add_argument("--input", required=True, help="Path to one image or a directory of images.")
    parser.add_argument("--output-dir", default="", help="Optional output directory override.")
    return parser.parse_args()


def iter_images(input_path: Path, allowed_exts: set[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = get_sensor_paths(config, args.session_name, args.sensor_id)
    calibration = load_json(paths.calibration_file)

    input_path = Path(args.input).resolve()
    undistort_cfg = config.get("undistort", {})
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths.undistort_dir
    ensure_dir(output_dir)

    allowed_exts = {ext.lower() for ext in undistort_cfg.get("image_extensions", [])}
    images = iter_images(input_path, allowed_exts)

    camera_matrix = np.array(calibration["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(calibration["dist_coeffs"], dtype=np.float64)
    optimal_camera_matrix = np.array(calibration["optimal_camera_matrix"], dtype=np.float64)
    roi = calibration["roi"]

    crop_to_roi = bool(undistort_cfg.get("crop_to_roi", True))
    save_side_by_side = bool(undistort_cfg.get("save_side_by_side", True))

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[WARN] Failed to read: {image_path}")
            continue

        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, optimal_camera_matrix)

        if crop_to_roi and roi["w"] > 0 and roi["h"] > 0:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            undistorted = undistorted[y : y + h, x : x + w]

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), undistorted)
        print(f"[DONE] Saved undistorted image: {output_path}")

        if save_side_by_side:
            side_by_side = build_side_by_side(image, undistorted)
            compare_path = output_dir / f"{image_path.stem}_compare{image_path.suffix}"
            cv2.imwrite(str(compare_path), side_by_side)
            print(f"[DONE] Saved comparison image: {compare_path}")


def build_side_by_side(original: np.ndarray, undistorted: np.ndarray) -> np.ndarray:
    target_height = min(original.shape[0], undistorted.shape[0])
    original_resized = resize_by_height(original, target_height)
    undistorted_resized = resize_by_height(undistorted, target_height)
    return cv2.hconcat([original_resized, undistorted_resized])


def resize_by_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
