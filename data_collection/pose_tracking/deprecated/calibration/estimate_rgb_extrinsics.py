from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "extrinsics.yaml"
DEFAULT_OUTPUT_NAME = "rgb_extrinsics_result"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate camera B pose in camera A coordinates using synchronized RGB checkerboard image pairs."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config file.")
    parser.add_argument("--session-name", default="", help="Session name under the output directory.")
    parser.add_argument("--camera-a-intrinsics", default="", help="Path to camera A RGB intrinsics JSON/YAML.")
    parser.add_argument("--camera-b-intrinsics", default="", help="Path to camera B RGB intrinsics JSON/YAML.")
    parser.add_argument("--rows", type=int, default=0, help="Checkerboard inner corner rows.")
    parser.add_argument("--cols", type=int, default=0, help="Checkerboard inner corner cols.")
    parser.add_argument("--square-size-mm", type=float, default=0.0, help="Checkerboard square size in millimeters.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Output filename stem written under the dataset directory.")
    parser.add_argument("--camera-a-frame", default="camera_a_color_optical_frame", help="Parent frame name for TF output.")
    parser.add_argument("--camera-b-frame", default="camera_b_color_optical_frame", help="Child frame name for TF output.")
    parser.add_argument("--min-valid-pairs", type=int, default=3, help="Minimum valid solved pairs required to emit a final result.")
    return parser.parse_args()


def resolve_path(path_value: str | Path, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    root = base_dir if base_dir is not None else SCRIPT_DIR
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
        return {}
    return load_yaml_or_json(config_path)


def apply_config_defaults(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    intrinsics_cfg = config.get("intrinsics", {})
    checkerboard_cfg = config.get("checkerboard", {})
    output_cfg = config.get("output", {})

    if not args.session_name:
        args.session_name = str(output_cfg.get("session_name", ""))
    if not args.camera_a_intrinsics:
        args.camera_a_intrinsics = str(intrinsics_cfg.get("camera_a", ""))
    if not args.camera_b_intrinsics:
        args.camera_b_intrinsics = str(intrinsics_cfg.get("camera_b", ""))
    if args.rows == 0:
        args.rows = int(checkerboard_cfg.get("rows", 0))
    if args.cols == 0:
        args.cols = int(checkerboard_cfg.get("cols", 0))
    if args.square_size_mm == 0.0:
        args.square_size_mm = float(checkerboard_cfg.get("square_size_mm", 0.0))
    if args.output_name == DEFAULT_OUTPUT_NAME:
        args.output_name = str(output_cfg.get("name", args.output_name))
    if args.camera_a_frame == "camera_a_color_optical_frame":
        args.camera_a_frame = str(output_cfg.get("camera_a_frame", args.camera_a_frame))
    if args.camera_b_frame == "camera_b_color_optical_frame":
        args.camera_b_frame = str(output_cfg.get("camera_b_frame", args.camera_b_frame))
    if args.min_valid_pairs == 3:
        args.min_valid_pairs = int(output_cfg.get("min_valid_pairs", args.min_valid_pairs))
    return args


def prompt_dataset_dir() -> str:
    return input("Enter session name: ").strip()


def create_object_points(rows: int, cols: int, square_size_mm: float) -> np.ndarray:
    object_points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid
    object_points *= square_size_mm
    return object_points


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, Optional[np.ndarray]]:
    success, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    if success and corners is not None:
        return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    success, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not success or corners is None:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


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
        raise RuntimeError(f"Checkerboard corners not found in image: {image_path.name}")

    pnp_success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    if not pnp_success:
        raise RuntimeError(f"solvePnP failed for image: {image_path.name}")

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
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


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


def extract_index(path: Path) -> Optional[int]:
    parts = path.stem.split("_")
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def list_common_pairs(dataset_dir: Path) -> list[tuple[int, Path, Path]]:
    camera_a_dir = dataset_dir / "camera_a"
    camera_b_dir = dataset_dir / "camera_b"
    map_a = {index: path for path in camera_a_dir.glob("*") if path.is_file() and (index := extract_index(path)) is not None}
    map_b = {index: path for path in camera_b_dir.glob("*") if path.is_file() and (index := extract_index(path)) is not None}
    common_indices = sorted(set(map_a.keys()) & set(map_b.keys()))
    return [(index, map_a[index], map_b[index]) for index in common_indices]


def parse_intrinsics_payload(payload: dict[str, Any], file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if "camera_matrix" in payload and "dist_coeffs" in payload:
        camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)
        return camera_matrix, dist_coeffs.reshape(-1, 1)

    if "cam0" in payload:
        camera_matrix = np.array(payload["cam0"].get("camera_matrix"), dtype=np.float64)
        dist_coeffs = np.array(payload["cam0"].get("dist_coeffs"), dtype=np.float64)
        return camera_matrix, dist_coeffs.reshape(-1, 1)

    raise ValueError(f"Unsupported intrinsics file format: {file_path}")


def load_intrinsics(path_value: str | Path, base_dir: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    path = resolve_path(path_value, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    payload = load_yaml_or_json(path)
    camera_matrix, dist_coeffs = parse_intrinsics_payload(payload, path)
    return path, camera_matrix, dist_coeffs


def write_outputs(dataset_dir: Path, output_name: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = dataset_dir / f"{output_name}.json"
    yaml_path = dataset_dir / f"{output_name}.yaml"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    return json_path, yaml_path


def validate_args(args: argparse.Namespace) -> None:
    if not args.session_name:
        raise ValueError("session-name is required.")
    if not args.camera_a_intrinsics:
        raise ValueError("camera-a-intrinsics is required.")
    if not args.camera_b_intrinsics:
        raise ValueError("camera-b-intrinsics is required.")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive.")
    if args.square_size_mm <= 0:
        raise ValueError("square-size-mm must be positive.")
    if args.min_valid_pairs < 1:
        raise ValueError("min-valid-pairs must be at least 1.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args = apply_config_defaults(args, config)
    if not args.session_name:
        args.session_name = prompt_dataset_dir()
    validate_args(args)

    config_path = resolve_path(args.config)
    dataset_dir = (DEFAULT_OUTPUT_ROOT / args.session_name).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    pairs = list_common_pairs(dataset_dir)
    if not pairs:
        raise RuntimeError(f"No paired images found under: {dataset_dir}")

    camera_a_intrinsics_path, camera_matrix_a, dist_coeffs_a = load_intrinsics(args.camera_a_intrinsics, config_path.parent)
    camera_b_intrinsics_path, camera_matrix_b, dist_coeffs_b = load_intrinsics(args.camera_b_intrinsics, config_path.parent)

    pattern_size = (args.cols, args.rows)
    object_points = create_object_points(args.rows, args.cols, args.square_size_mm)

    pair_results: list[dict[str, Any]] = []
    quaternions: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    reproj_a_all: list[float] = []
    reproj_b_all: list[float] = []

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

        quaternions.append(quaternion_ab)
        translations.append(translation_ab)
        reproj_a_all.append(reproj_a)
        reproj_b_all.append(reproj_b)

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
                "rotation_matrix_a_b": rotation_ab.tolist(),
                "translation_vector_a_b": translation_ab.tolist(),
                "quaternion_xyzw_a_b": quaternion_ab.tolist(),
            }
        )

    if len(pair_results) < args.min_valid_pairs:
        raise RuntimeError(
            f"Only {len(pair_results)} valid pairs were solved, but at least {args.min_valid_pairs} are required."
        )

    mean_translation = np.mean(np.stack(translations, axis=0), axis=0)
    mean_quaternion = average_quaternions(quaternions)
    mean_rotation = quaternion_to_rotation_matrix(mean_quaternion)
    final_transform = to_homogeneous(mean_rotation, mean_translation)

    payload = {
        "dataset_dir": str(dataset_dir),
        "checkerboard": {
            "rows": args.rows,
            "cols": args.cols,
            "square_size_mm": args.square_size_mm,
        },
        "camera_a_intrinsics_file": str(camera_a_intrinsics_path),
        "camera_b_intrinsics_file": str(camera_b_intrinsics_path),
        "num_input_pairs": len(pairs),
        "num_valid_pairs": len(pair_results),
        "mean_camera_a_reprojection_error": float(np.mean(reproj_a_all)),
        "mean_camera_b_reprojection_error": float(np.mean(reproj_b_all)),
        "pair_results": pair_results,
        "final_extrinsics": {
            "rotation_matrix_a_b": mean_rotation.tolist(),
            "translation_vector_a_b": mean_translation.tolist(),
            "quaternion_xyzw_a_b": mean_quaternion.tolist(),
            "transform_a_b": final_transform.tolist(),
        },
        "tf_static": {
            "parent_frame": args.camera_a_frame,
            "child_frame": args.camera_b_frame,
            "translation_xyz": mean_translation.tolist(),
            "quaternion_xyzw": mean_quaternion.tolist(),
        },
    }

    json_path, yaml_path = write_outputs(dataset_dir, args.output_name, payload)
    print(f"[DONE] Wrote JSON result to: {json_path}")
    print(f"[DONE] Wrote YAML result to: {yaml_path}")


if __name__ == "__main__":
    main()
