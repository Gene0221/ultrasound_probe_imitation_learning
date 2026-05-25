from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class SensorPaths:
    sensor_id: str
    image_dir: Path
    output_dir: Path
    calibration_file: Path
    corners_vis_dir: Path
    undistort_dir: Path


@dataclass
class PairPaths:
    pair_id: str
    output_dir: Path
    calibration_file: Path
    corners_vis_dir: Path


@dataclass
class SessionRoots:
    raw_root: Path
    output_root: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping.")

    data["_config_path"] = str(config_file)
    data["_config_dir"] = config_file.parent
    return data


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_session_roots(config: dict[str, Any], session_name: str) -> SessionRoots:
    config_dir = Path(config["_config_dir"])
    project_cfg = config.get("project", {})
    raw_root = resolve_path(config_dir, project_cfg.get("raw_root", "./data/raw")) / session_name
    output_root = resolve_path(config_dir, project_cfg.get("output_root", "./data/output")) / session_name
    return SessionRoots(raw_root=raw_root, output_root=output_root)


def get_sensor_ids(config: dict[str, Any]) -> list[str]:
    sensors = config.get("sensors", {})
    return [sensor_id for sensor_id, sensor_cfg in sensors.items() if sensor_cfg.get("enabled", True)]


def get_sensor_config(config: dict[str, Any], sensor_id: str) -> dict[str, Any]:
    sensors = config.get("sensors", {})
    if sensor_id not in sensors:
        raise KeyError(f"Sensor '{sensor_id}' is not defined in config.")
    return sensors[sensor_id]


def get_pair_config(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    pairs = config.get("pairs", {})
    if pair_id not in pairs:
        raise KeyError(f"Pair '{pair_id}' is not defined in config.")
    return pairs[pair_id]


def get_sensor_paths(config: dict[str, Any], session_name: str, sensor_id: str) -> SensorPaths:
    session_roots = get_session_roots(config, session_name)
    sensor_cfg = get_sensor_config(config, sensor_id)
    mono_cfg = config.get("mono_calibration", {})
    undistort_cfg = config.get("undistort", {})
    sensor_dirname = sensor_cfg.get("dirname", sensor_id)

    image_dir = session_roots.raw_root / sensor_dirname
    output_dir = session_roots.output_root / sensor_dirname
    corners_vis_dir = output_dir / mono_cfg.get("visualization_dirname", "corners_vis")
    undistort_dir = output_dir / undistort_cfg.get("output_dirname", "undistorted")
    calibration_file = output_dir / "calibration_result.json"

    return SensorPaths(
        sensor_id=sensor_id,
        image_dir=image_dir,
        output_dir=output_dir,
        calibration_file=calibration_file,
        corners_vis_dir=corners_vis_dir,
        undistort_dir=undistort_dir,
    )


def get_pair_paths(config: dict[str, Any], session_name: str, pair_id: str) -> PairPaths:
    session_roots = get_session_roots(config, session_name)
    pair_cfg = get_pair_config(config, pair_id)
    stereo_cfg = config.get("stereo_calibration", {})
    output_dir = session_roots.output_root / pair_cfg.get("output_dirname", pair_id)
    calibration_file = output_dir / pair_cfg.get("result_filename", "stereo_calibration_result.json")
    corners_vis_dir = output_dir / stereo_cfg.get("visualization_dirname", "corners_vis")
    return PairPaths(pair_id=pair_id, output_dir=output_dir, calibration_file=calibration_file, corners_vis_dir=corners_vis_dir)


def get_bundle_file(config: dict[str, Any], session_name: str) -> Path:
    session_roots = get_session_roots(config, session_name)
    bundle_cfg = config.get("bundle", {})
    bundle_dir = session_roots.output_root / bundle_cfg.get("output_dirname", "bundle")
    return bundle_dir / bundle_cfg.get("filename", "device_calibration_bundle.json")


def list_sensor_images(config: dict[str, Any], session_name: str, sensor_id: str) -> list[Path]:
    sensor_cfg = get_sensor_config(config, sensor_id)
    paths = get_sensor_paths(config, session_name, sensor_id)
    pattern = sensor_cfg.get("image_pattern", f"{sensor_id}_*.png")
    if not paths.image_dir.exists():
        return []
    images = sorted(paths.image_dir.glob(pattern))
    return [p for p in images if p.is_file()]


def build_capture_filename(sensor_id: str, frame_index: int, extension: str) -> str:
    return f"{sensor_id}_{frame_index:04d}{extension}"


def extract_frame_index(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def map_images_by_index(images: list[Path]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for image_path in images:
        frame_index = extract_frame_index(image_path)
        if frame_index is not None:
            mapping[frame_index] = image_path
    return mapping


def list_common_frame_sets(
    config: dict[str, Any],
    session_name: str,
    sensor_ids: list[str],
) -> list[tuple[int, dict[str, Path]]]:
    if not sensor_ids:
        return []

    sensor_maps = {
        sensor_id: map_images_by_index(list_sensor_images(config, session_name, sensor_id))
        for sensor_id in sensor_ids
    }
    common_indices = set.intersection(*(set(mapping.keys()) for mapping in sensor_maps.values())) if sensor_maps else set()
    frame_sets: list[tuple[int, dict[str, Path]]] = []
    for frame_index in sorted(common_indices):
        frame_sets.append((frame_index, {sensor_id: sensor_maps[sensor_id][frame_index] for sensor_id in sensor_ids}))
    return frame_sets


def count_complete_frame_sets(config: dict[str, Any], session_name: str, sensor_ids: list[str]) -> int:
    return len(list_common_frame_sets(config, session_name, sensor_ids))


def get_next_frame_index(config: dict[str, Any], session_name: str, sensor_ids: list[str]) -> int:
    max_index = 0
    for sensor_id in sensor_ids:
        for image_path in list_sensor_images(config, session_name, sensor_id):
            frame_index = extract_frame_index(image_path)
            if frame_index is not None:
                max_index = max(max_index, frame_index)
    return max_index + 1


def prompt_session_name(default_name: str) -> str:
    while True:
        session_name = input(f"Enter save name for this calibration session [{default_name}]: ").strip()
        if not session_name:
            session_name = default_name
        session_name = sanitize_session_name(session_name)
        if session_name:
            return session_name
        print("[WARN] Save name cannot be empty.")


def sanitize_session_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def prompt_existing_session_strategy(session_name: str) -> str:
    while True:
        choice = input(
            f"Session '{session_name}' already exists. Choose overwrite / merge / cancel [merge]: "
        ).strip().lower()
        if not choice:
            return "merge"
        if choice in {"overwrite", "merge", "cancel"}:
            return choice
        print("[WARN] Please enter overwrite, merge, or cancel.")


def prepare_session_storage(config: dict[str, Any], session_name: str) -> str:
    session_roots = get_session_roots(config, session_name)
    raw_exists = session_roots.raw_root.exists()
    output_exists = session_roots.output_root.exists()

    strategy = "new"
    if raw_exists or output_exists:
        strategy = prompt_existing_session_strategy(session_name)
        if strategy == "cancel":
            raise SystemExit("Session creation cancelled by user.")
        if strategy == "overwrite":
            remove_dir_if_exists(session_roots.raw_root)
            remove_dir_if_exists(session_roots.output_root)

    ensure_dir(session_roots.raw_root)
    ensure_dir(session_roots.output_root)
    return strategy


def create_object_points(config: dict[str, Any]) -> np.ndarray:
    board_cfg = config["board"]
    cols, rows = board_cfg["pattern_size"]
    square_size = float(board_cfg["square_size_mm"])

    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size
    return objp


def get_find_chessboard_flags(config: dict[str, Any]) -> int:
    corner_cfg = config.get("corner_detection", {})
    flags = 0
    if corner_cfg.get("adaptive_thresh", True):
        flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
    if corner_cfg.get("normalize_image", True):
        flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    if corner_cfg.get("fast_check", False):
        flags |= cv2.CALIB_CB_FAST_CHECK
    return flags


def find_corners(gray: np.ndarray, config: dict[str, Any]) -> tuple[bool, np.ndarray | None]:
    board_cfg = config["board"]
    pattern_size = tuple(board_cfg["pattern_size"])
    corner_cfg = config.get("corner_detection", {})
    use_sb = corner_cfg.get("use_find_chessboard_sb", True)

    if use_sb:
        success, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    else:
        flags = get_find_chessboard_flags(config)
        success, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not success or corners is None:
        return False, None

    subpix_cfg = corner_cfg.get("subpix", {})
    if subpix_cfg.get("enabled", True) and not use_sb:
        win_size = tuple(subpix_cfg.get("win_size", [11, 11]))
        zero_zone = tuple(subpix_cfg.get("zero_zone", [-1, -1]))
        criteria_cfg = subpix_cfg.get("criteria", {})
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(criteria_cfg.get("max_iter", 30)),
            float(criteria_cfg.get("epsilon", 0.001)),
        )
        corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)

    return True, corners


def detect_corners_in_image(image: np.ndarray, config: dict[str, Any]) -> tuple[bool, np.ndarray | None]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return find_corners(gray, config)


def compute_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, list[float]]:
    per_image_errors: list[float] = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        reprojected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, reprojected, cv2.NORM_L2) / len(reprojected)
        per_image_errors.append(float(error))
    mean_error = float(sum(per_image_errors) / len(per_image_errors)) if per_image_errors else 0.0
    return mean_error, per_image_errors


def collect_single_camera_observations(
    config: dict[str, Any],
    session_name: str,
    sensor_id: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], tuple[int, int]]:
    paths = get_sensor_paths(config, session_name, sensor_id)
    images = list_sensor_images(config, session_name, sensor_id)
    min_images = int(config.get("mono_calibration", {}).get("min_images", 10))
    if len(images) < min_images:
        raise RuntimeError(
            f"Sensor '{sensor_id}' only has {len(images)} images, but config requires at least {min_images}."
        )

    object_points_template = create_object_points(config)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    image_size: tuple[int, int] | None = None

    save_vis = bool(config.get("mono_calibration", {}).get("save_visualizations", True))
    if save_vis:
        ensure_dir(paths.corners_vis_dir)

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[WARN] Failed to read: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        success, corners = find_corners(gray, config)
        if not success or corners is None:
            print(f"[INFO] Corners not found: {image_path.name}")
            continue

        object_points.append(object_points_template.copy())
        image_points.append(corners)
        used_images.append(image_path.name)
        print(f"[OK] Corners detected: {image_path.name}")

        if save_vis:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
            cv2.drawChessboardCorners(vis, tuple(config["board"]["pattern_size"]), corners, True)
            cv2.imwrite(str(paths.corners_vis_dir / image_path.name), vis)

    if image_size is None:
        raise RuntimeError(f"No valid images found for sensor '{sensor_id}'.")
    if len(image_points) < min_images:
        raise RuntimeError(
            f"Sensor '{sensor_id}' only has {len(image_points)} valid corner detections, "
            f"but config requires at least {min_images}."
        )

    return object_points, image_points, used_images, image_size


def run_mono_calibration(
    config: dict[str, Any],
    session_name: str,
    sensor_id: str,
) -> dict[str, Any]:
    paths = get_sensor_paths(config, session_name, sensor_id)
    images = list_sensor_images(config, session_name, sensor_id)
    object_points, image_points, used_images, image_size = collect_single_camera_observations(
        config, session_name, sensor_id
    )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    alpha = float(config.get("mono_calibration", {}).get("undistort_alpha", 0.0))
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha,
        image_size,
    )
    mean_error, per_image_errors = compute_reprojection_error(
        object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    result = {
        "session_name": session_name,
        "sensor_id": sensor_id,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "board": config["board"],
        "num_input_images": len(images),
        "num_used_images": len(used_images),
        "used_images": used_images,
        "rms": float(rms),
        "mean_reprojection_error": mean_error,
        "per_image_reprojection_error": per_image_errors,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "optimal_camera_matrix": optimal_camera_matrix.tolist(),
        "roi": {"x": int(roi[0]), "y": int(roi[1]), "w": int(roi[2]), "h": int(roi[3])},
        "rvecs": [rvec.tolist() for rvec in rvecs],
        "tvecs": [tvec.tolist() for tvec in tvecs],
    }

    ensure_dir(paths.output_dir)
    save_json(paths.calibration_file, result)
    return result


def collect_pair_observations(
    config: dict[str, Any],
    session_name: str,
    left_sensor_id: str,
    right_sensor_id: str,
    corners_vis_dir: Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], tuple[int, int]]:
    frame_sets = list_common_frame_sets(config, session_name, [left_sensor_id, right_sensor_id])
    min_pairs = int(config.get("stereo_calibration", {}).get("min_pairs", 10))
    if len(frame_sets) < min_pairs:
        raise RuntimeError(
            f"Only found {len(frame_sets)} common frame sets for pair '{left_sensor_id}-{right_sensor_id}', "
            f"but config requires at least {min_pairs}."
        )

    object_points_template = create_object_points(config)
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    used_indices: list[int] = []
    image_size: tuple[int, int] | None = None

    save_vis = bool(config.get("stereo_calibration", {}).get("save_visualizations", True))
    if save_vis and corners_vis_dir is not None:
        ensure_dir(corners_vis_dir)

    for frame_index, image_map in frame_sets:
        left_image = cv2.imread(str(image_map[left_sensor_id]), cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(str(image_map[right_sensor_id]), cv2.IMREAD_UNCHANGED)
        if left_image is None or right_image is None:
            print(f"[WARN] Failed to read frame set {frame_index:04d}")
            continue

        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if left_image.ndim == 3 else left_image
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if right_image.ndim == 3 else right_image

        if image_size is None:
            image_size = (left_gray.shape[1], left_gray.shape[0])

        if (left_gray.shape[1], left_gray.shape[0]) != image_size or (right_gray.shape[1], right_gray.shape[0]) != image_size:
            print(f"[WARN] Skipping frame {frame_index:04d} due to inconsistent image size.")
            continue

        left_success, left_corners = find_corners(left_gray, config)
        right_success, right_corners = find_corners(right_gray, config)
        if not left_success or left_corners is None or not right_success or right_corners is None:
            print(f"[INFO] Corners not found in both images for frame {frame_index:04d}")
            continue

        object_points.append(object_points_template.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)
        used_indices.append(frame_index)
        print(f"[OK] Stereo corners detected: frame {frame_index:04d}")

        if save_vis and corners_vis_dir is not None:
            left_vis = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR) if left_image.ndim == 2 else left_image.copy()
            right_vis = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR) if right_image.ndim == 2 else right_image.copy()
            cv2.drawChessboardCorners(left_vis, tuple(config["board"]["pattern_size"]), left_corners, True)
            cv2.drawChessboardCorners(right_vis, tuple(config["board"]["pattern_size"]), right_corners, True)
            cv2.imwrite(str(corners_vis_dir / f"{left_sensor_id}_{frame_index:04d}.png"), left_vis)
            cv2.imwrite(str(corners_vis_dir / f"{right_sensor_id}_{frame_index:04d}.png"), right_vis)

    if image_size is None:
        raise RuntimeError(f"No valid image size could be determined for pair '{left_sensor_id}-{right_sensor_id}'.")
    if len(used_indices) < min_pairs:
        raise RuntimeError(
            f"Only {len(used_indices)} valid synchronized detections for pair '{left_sensor_id}-{right_sensor_id}', "
            f"but config requires at least {min_pairs}."
        )

    return object_points, left_points, right_points, used_indices, image_size


def _stereo_criteria(config: dict[str, Any]) -> tuple[int, int, float]:
    criteria_cfg = config.get("stereo_calibration", {}).get("criteria", {})
    return (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(criteria_cfg.get("max_iter", 100)),
        float(criteria_cfg.get("epsilon", 1e-5)),
    )


def run_pair_calibration(
    config: dict[str, Any],
    session_name: str,
    pair_id: str,
    left_calibration: dict[str, Any],
    right_calibration: dict[str, Any],
) -> dict[str, Any]:
    pair_cfg = get_pair_config(config, pair_id)
    left_sensor_id = pair_cfg["left_sensor_id"]
    right_sensor_id = pair_cfg["right_sensor_id"]
    pair_paths = get_pair_paths(config, session_name, pair_id)

    object_points, left_points, right_points, used_indices, image_size = collect_pair_observations(
        config,
        session_name,
        left_sensor_id,
        right_sensor_id,
        corners_vis_dir=pair_paths.corners_vis_dir,
    )

    left_camera_matrix = np.array(left_calibration["camera_matrix"], dtype=np.float64)
    left_dist_coeffs = np.array(left_calibration["dist_coeffs"], dtype=np.float64)
    right_camera_matrix = np.array(right_calibration["camera_matrix"], dtype=np.float64)
    right_dist_coeffs = np.array(right_calibration["dist_coeffs"], dtype=np.float64)

    flags = cv2.CALIB_FIX_INTRINSIC
    stereo_rms, _, _, _, _, rotation_matrix, translation_vector, essential_matrix, fundamental_matrix = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        image_size,
        criteria=_stereo_criteria(config),
        flags=flags,
    )

    rectification = None
    if bool(pair_cfg.get("compute_rectification", True)):
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
            image_size,
            rotation_matrix,
            translation_vector,
            alpha=float(config.get("stereo_calibration", {}).get("rectify_alpha", 0.0)),
        )
        rectification = {
            "R1": r1.tolist(),
            "R2": r2.tolist(),
            "P1": p1.tolist(),
            "P2": p2.tolist(),
            "Q": q.tolist(),
            "roi1": {"x": int(roi1[0]), "y": int(roi1[1]), "w": int(roi1[2]), "h": int(roi1[3])},
            "roi2": {"x": int(roi2[0]), "y": int(roi2[1]), "w": int(roi2[2]), "h": int(roi2[3])},
        }

    result = {
        "session_name": session_name,
        "pair_id": pair_id,
        "left_sensor_id": left_sensor_id,
        "right_sensor_id": right_sensor_id,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "board": config["board"],
        "num_used_pairs": len(used_indices),
        "used_frame_indices": used_indices,
        "stereo_rms": float(stereo_rms),
        "left_camera_matrix": left_camera_matrix.tolist(),
        "left_dist_coeffs": left_dist_coeffs.tolist(),
        "right_camera_matrix": right_camera_matrix.tolist(),
        "right_dist_coeffs": right_dist_coeffs.tolist(),
        "rotation_matrix": rotation_matrix.tolist(),
        "translation_vector": translation_vector.tolist(),
        "essential_matrix": essential_matrix.tolist(),
        "fundamental_matrix": fundamental_matrix.tolist(),
        "rectification": rectification,
    }

    ensure_dir(pair_paths.output_dir)
    save_json(pair_paths.calibration_file, result)
    return result


def summarize_bundle(
    config: dict[str, Any],
    session_name: str,
    mono_results: dict[str, dict[str, Any]],
    pair_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_sensor = config.get("reference_frames", {}).get("primary_sensor", "ir_left")
    bundle = {
        "session_name": session_name,
        "reference_sensor": reference_sensor,
        "board": config["board"],
        "sensors": mono_results,
        "pairs": pair_results,
    }
    bundle_file = get_bundle_file(config, session_name)
    save_json(bundle_file, bundle)
    return bundle
