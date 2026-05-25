from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ultrasound_il.pose import normalize_pose_delta_vector
from ultrasound_il.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a toy dataset for continuous policy pipeline debugging.")
    parser.add_argument("--output-dir", type=str, required=True, help="Dataset output directory.")
    parser.add_argument("--num-train-traj", type=int, default=4)
    parser.add_argument("--num-val-traj", type=int, default=2)
    parser.add_argument("--frames-per-traj", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-force", action="store_true")
    parser.add_argument("--force-dim", type=int, default=6)
    return parser.parse_args()


def random_pose_delta_7d() -> list[float]:
    translation = np.random.uniform(-0.01, 0.01, size=3).astype(np.float32)
    axis = np.random.normal(size=3).astype(np.float32)
    axis_norm = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32) if axis_norm < 1e-8 else axis / axis_norm
    angle = random.uniform(-0.1, 0.1)
    sin_half = math.sin(angle / 2.0)
    cos_half = math.cos(angle / 2.0)
    quaternion = np.array(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half],
        dtype=np.float32,
    )
    return normalize_pose_delta_vector(np.concatenate([translation, quaternion], axis=0)).tolist()


def random_force(force_dim: int) -> list[float]:
    return np.random.uniform(-3.0, 3.0, size=force_dim).astype(np.float32).tolist()


def synthesize_ultrasound_like_image(image_size: int, frame_idx: int, pose_delta: list[float]) -> np.ndarray:
    canvas = np.random.normal(loc=25, scale=10, size=(image_size, image_size)).astype(np.float32)
    center_x = image_size // 2 + int(24 * math.sin(frame_idx / 3.0 + pose_delta[0] * 50))
    center_y = image_size // 2 + int(24 * math.cos(frame_idx / 4.0 + pose_delta[1] * 50))
    radius = 20 + int(abs(pose_delta[2]) * 1000) % 12
    yy, xx = np.ogrid[:image_size, :image_size]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
    canvas[mask] += 140
    return np.clip(canvas, 0, 255).astype(np.uint8)


def build_trajectory(
    root_dir: Path,
    dataset_root: Path,
    trajectory_id: str,
    frames_per_traj: int,
    image_size: int,
    use_force: bool,
    force_dim: int,
) -> dict:
    images_dir = ensure_dir(root_dir / "images")
    frames = []
    for frame_idx in range(frames_per_traj):
        pose_delta = random_pose_delta_7d()
        image = synthesize_ultrasound_like_image(image_size, frame_idx, pose_delta)
        filename = f"{frame_idx:06d}.png"
        Image.fromarray(image, mode="L").save(images_dir / filename)
        frame_payload: dict[str, object] = {
            "image": f"images/{filename}",
            "pose_delta_7d": pose_delta,
        }
        if use_force:
            frame_payload["force"] = random_force(force_dim)
        frames.append(frame_payload)

    metadata = {"trajectory_id": trajectory_id, "frames": frames}
    save_json(metadata, root_dir / "metadata.json")
    return {
        "trajectory_id": trajectory_id,
        "root_dir": str(root_dir.relative_to(dataset_root)).replace("\\", "/"),
        "metadata_path": str((root_dir / "metadata.json").relative_to(dataset_root)).replace("\\", "/"),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    trajectories_root = ensure_dir(output_dir / "trajectories")
    manifests_root = ensure_dir(output_dir / "manifests")

    train_trajectories = []
    for idx in range(args.num_train_traj):
        trajectory_id = f"train_traj_{idx:03d}"
        root_dir = ensure_dir(trajectories_root / trajectory_id)
        train_trajectories.append(
            build_trajectory(
                root_dir=root_dir,
                dataset_root=output_dir,
                trajectory_id=trajectory_id,
                frames_per_traj=args.frames_per_traj,
                image_size=args.image_size,
                use_force=args.use_force,
                force_dim=args.force_dim,
            )
        )

    val_trajectories = []
    for idx in range(args.num_val_traj):
        trajectory_id = f"val_traj_{idx:03d}"
        root_dir = ensure_dir(trajectories_root / trajectory_id)
        val_trajectories.append(
            build_trajectory(
                root_dir=root_dir,
                dataset_root=output_dir,
                trajectory_id=trajectory_id,
                frames_per_traj=args.frames_per_traj,
                image_size=args.image_size,
                use_force=args.use_force,
                force_dim=args.force_dim,
            )
        )

    manifest_payload = {
        "use_force": args.use_force,
        "force_dim": args.force_dim,
    }
    save_json({**manifest_payload, "trajectories": train_trajectories}, manifests_root / "train_manifest.json")
    save_json({**manifest_payload, "trajectories": val_trajectories}, manifests_root / "val_manifest.json")
    print(f"Toy dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
