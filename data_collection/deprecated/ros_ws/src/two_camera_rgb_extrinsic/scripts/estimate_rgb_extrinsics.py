#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate camera B pose in camera A coordinates using paired RGB checkerboard observations.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory containing camera_a/ and camera_b/ image folders.")
    parser.add_argument("--camera-a-intrinsics", required=True, help="Path to camera A RGB intrinsics JSON.")
    parser.add_argument("--camera-b-intrinsics", required=True, help="Path to camera B RGB intrinsics JSON.")
    parser.add_argument("--rows", type=int, required=True, help="Number of checkerboard inner corners along rows.")
    parser.add_argument("--cols", type=int, required=True, help="Number of checkerboard inner corners along columns.")
    parser.add_argument("--square-size", type=float, required=True, help="Checkerboard square size in millimeters.")
    parser.add_argument("--output-prefix", default="extrinsics_result", help="Output filename prefix inside the dataset directory.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def create_object_points(rows: int, cols: int, square_size_mm: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size_mm
    return objp


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    success, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    if success and corners is not None:
        return True, corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    success, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not success or corners is None:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def solve_board_pose(
    image_path: Path,
    pattern_size: tuple[int, int],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    success, corners = find_corners(gray, pattern_size)
    if not success or corners is None:
        raise RuntimeError(f"Checkerboard corners not found in image: {image_path}")

    pnp_success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    if not pnp_success:
        raise RuntimeError(f"solvePnP failed for image: {image_path}")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error = cv2.norm(corners, projected, cv2.NORM_L2) / len(projected)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return rotation_matrix, tvec.reshape(3), float(reprojection_error)


def to_homogeneous(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector.reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


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


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quaternion
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def average_quaternions(quaternions: list[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros(4, dtype=np.float64)
    reference = quaternions[0]
    for quaternion in quaternions:
        q = quaternion.copy()
        if np.dot(reference, q) < 0:
            q = -q
        accumulator += q
    accumulator /= np.linalg.norm(accumulator)
    return accumulator


def list_common_pairs(dataset_dir: Path) -> list[tuple[int, Path, Path]]:
    camera_a_dir = dataset_dir / "camera_a"
    camera_b_dir = dataset_dir / "camera_b"
    map_a = {extract_index(path): path for path in camera_a_dir.glob("*") if path.is_file() and extract_index(path) is not None}
    map_b = {extract_index(path): path for path in camera_b_dir.glob("*") if path.is_file() and extract_index(path) is not None}
    common_indices = sorted(set(map_a.keys()) & set(map_b.keys()))
    return [(index, map_a[index], map_b[index]) for index in common_indices]


def extract_index(path: Path) -> int | None:
    stem_parts = path.stem.split("_")
    if not stem_parts:
        return None
    try:
        return int(stem_parts[-1])
    except ValueError:
        return None


def write_outputs(dataset_dir: Path, output_prefix: str, payload: dict[str, Any]) -> None:
    json_path = dataset_dir / f"{output_prefix}.json"
    yaml_path = dataset_dir / f"{output_prefix}.yaml"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[DONE] Wrote JSON result to: {json_path}")
    print(f"[DONE] Wrote YAML result to: {yaml_path}")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    pairs = list_common_pairs(dataset_dir)
    if not pairs:
        raise RuntimeError(f"No paired images found in dataset directory: {dataset_dir}")

    camera_a_intrinsics = load_json(args.camera_a_intrinsics)
    camera_b_intrinsics = load_json(args.camera_b_intrinsics)
    camera_matrix_a = np.array(camera_a_intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs_a = np.array(camera_a_intrinsics["dist_coeffs"], dtype=np.float64)
    camera_matrix_b = np.array(camera_b_intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs_b = np.array(camera_b_intrinsics["dist_coeffs"], dtype=np.float64)

    pattern_size = (args.cols, args.rows)
    object_points = create_object_points(args.rows, args.cols, args.square_size)

    pair_results: list[dict[str, Any]] = []
    transforms_ab: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    translations: list[np.ndarray] = []

    for pair_index, image_a_path, image_b_path in pairs:
        try:
            rotation_a, translation_a, reproj_a = solve_board_pose(
                image_a_path, pattern_size, object_points, camera_matrix_a, dist_coeffs_a
            )
            rotation_b, translation_b, reproj_b = solve_board_pose(
                image_b_path, pattern_size, object_points, camera_matrix_b, dist_coeffs_b
            )
        except RuntimeError as exc:
            print(f"[WARN] Skipping pair {pair_index:04d}: {exc}")
            continue

        transform_a_board = to_homogeneous(rotation_a, translation_a)
        transform_b_board = to_homogeneous(rotation_b, translation_b)
        transform_a_b = transform_a_board @ invert_transform(transform_b_board)
        rotation_ab = transform_a_b[:3, :3]
        translation_ab = transform_a_b[:3, 3]
        quaternion_ab = rotation_matrix_to_quaternion(rotation_ab)

        transforms_ab.append(transform_a_b)
        quaternions.append(quaternion_ab)
        translations.append(translation_ab)

        pair_results.append(
            {
                "pair_index": pair_index,
                "camera_a_image": image_a_path.name,
                "camera_b_image": image_b_path.name,
                "camera_a_reprojection_error": reproj_a,
                "camera_b_reprojection_error": reproj_b,
                "camera_a_to_board": transform_a_board.tolist(),
                "camera_b_to_board": transform_b_board.tolist(),
                "transform_a_b": transform_a_b.tolist(),
                "rotation_a_b": rotation_ab.tolist(),
                "translation_a_b": translation_ab.tolist(),
                "quaternion_xyzw_a_b": quaternion_ab.tolist(),
            }
        )

    if not pair_results:
        raise RuntimeError("No valid checkerboard pairs were successfully solved.")

    mean_translation = np.mean(np.stack(translations, axis=0), axis=0)
    mean_quaternion = average_quaternions(quaternions)
    mean_rotation = quaternion_to_rotation_matrix(mean_quaternion)
    final_transform = to_homogeneous(mean_rotation, mean_translation)

    payload = {
        "dataset_dir": str(dataset_dir),
        "checkerboard": {
            "rows": args.rows,
            "cols": args.cols,
            "square_size_mm": args.square_size,
        },
        "camera_a_intrinsics_file": str(Path(args.camera_a_intrinsics).resolve()),
        "camera_b_intrinsics_file": str(Path(args.camera_b_intrinsics).resolve()),
        "num_input_pairs": len(pairs),
        "num_valid_pairs": len(pair_results),
        "pair_results": pair_results,
        "final_extrinsics": {
            "rotation_matrix_a_b": mean_rotation.tolist(),
            "translation_vector_a_b": mean_translation.tolist(),
            "quaternion_xyzw_a_b": mean_quaternion.tolist(),
            "transform_a_b": final_transform.tolist(),
        },
        "tf_static": {
            "parent_frame": "camera_a_color_optical_frame",
            "child_frame": "camera_b_color_optical_frame",
            "translation_xyz": mean_translation.tolist(),
            "quaternion_xyzw": mean_quaternion.tolist(),
        },
    }
    write_outputs(dataset_dir, args.output_prefix, payload)


if __name__ == "__main__":
    main()
