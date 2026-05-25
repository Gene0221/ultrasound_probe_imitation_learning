from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .pose import normalize_pose_delta_vector, pose_representation_dim, pose_sequence_pad_value
from .utils import load_json


@dataclass
class FrameRecord:
    image_path: Path
    pose_delta: Any | None
    force: Any | None


@dataclass
class WindowSample:
    trajectory_id: str
    frame_records: list[FrameRecord]
    target_pose_delta: np.ndarray


def load_manifest_samples(
    manifest_path: str | Path,
    window_size: int,
    pose_representation: str,
    use_force: bool,
    force_dim: int,
) -> list[WindowSample]:
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    samples: list[WindowSample] = []

    for trajectory in manifest["trajectories"]:
        root_dir = _resolve_manifest_relative_path(manifest_path, trajectory["root_dir"])
        metadata_path = _resolve_manifest_relative_path(manifest_path, trajectory["metadata_path"])
        metadata = load_json(metadata_path)
        frames = metadata["frames"]
        trajectory_id = metadata.get("trajectory_id", trajectory["trajectory_id"])

        if len(frames) < window_size:
            continue

        for end_idx in range(window_size - 1, len(frames)):
            current_window: list[FrameRecord] = []
            valid = True
            for frame_idx in range(end_idx - window_size + 1, end_idx + 1):
                frame = frames[frame_idx]
                pose_delta = _extract_pose_delta(frame)
                if frame_idx == end_idx and pose_delta is None:
                    valid = False
                    break
                force = _extract_force(frame, force_dim=force_dim) if use_force else None
                current_window.append(
                    FrameRecord(
                        image_path=(root_dir / frame["image"]).resolve(),
                        pose_delta=pose_delta,
                        force=force,
                    )
                )

            if not valid:
                continue

            target_pose_delta = normalize_pose_delta_vector(
                _extract_pose_delta(frames[end_idx]),
                representation=pose_representation,
            )
            samples.append(
                WindowSample(
                    trajectory_id=trajectory_id,
                    frame_records=current_window,
                    target_pose_delta=target_pose_delta,
                )
            )

    return samples


def _resolve_manifest_relative_path(manifest_path: Path, relative_or_absolute: str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate.resolve()

    direct = (manifest_path.parent / candidate).resolve()
    if direct.exists():
        return direct

    return (manifest_path.parent.parent / candidate).resolve()


def _extract_pose_delta(frame_payload: dict[str, Any]) -> Any | None:
    if "pose_delta_7d" in frame_payload:
        return frame_payload["pose_delta_7d"]
    if "pose_delta" in frame_payload:
        return frame_payload["pose_delta"]
    return None


def _extract_force(frame_payload: dict[str, Any], force_dim: int) -> np.ndarray:
    if "force" not in frame_payload:
        return np.zeros(force_dim, dtype=np.float32)

    force = np.asarray(frame_payload["force"], dtype=np.float32).reshape(-1)
    if force.shape[0] != force_dim:
        raise ValueError(f"Expected force dim {force_dim}, got {force.shape[0]}.")
    return force


class TemporalPolicyDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_transform,
        window_size: int = 8,
        pose_representation: str = "pose7_quaternion",
        use_force: bool = True,
        force_dim: int = 6,
    ) -> None:
        self.samples = load_manifest_samples(
            manifest_path=manifest_path,
            window_size=window_size,
            pose_representation=pose_representation,
            use_force=use_force,
            force_dim=force_dim,
        )
        self.image_transform = image_transform
        self.window_size = window_size
        self.pose_representation = pose_representation
        self.pose_feature_dim = pose_representation_dim(pose_representation)
        self.use_force = use_force
        self.force_dim = force_dim

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = Image.merge("RGB", (image, image, image))
        return self.image_transform(image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_tensors = [self._load_rgb_image(record.image_path) for record in sample.frame_records]
        history_pose_deltas: list[np.ndarray] = []
        force_sequence: list[np.ndarray] = []

        zero_pose = pose_sequence_pad_value(self.pose_representation)
        zero_force = np.zeros(self.force_dim, dtype=np.float32)

        for step_idx, record in enumerate(sample.frame_records):
            if step_idx == len(sample.frame_records) - 1:
                history_pose_deltas.append(zero_pose)
            elif record.pose_delta is None:
                raise ValueError(
                    f"Missing historical pose_delta in trajectory {sample.trajectory_id} at sample index {index}."
                )
            else:
                history_pose_deltas.append(
                    normalize_pose_delta_vector(record.pose_delta, representation=self.pose_representation)
                )

            if self.use_force:
                force_sequence.append(
                    np.asarray(record.force, dtype=np.float32).reshape(self.force_dim)
                    if record.force is not None
                    else zero_force
                )
            else:
                force_sequence.append(zero_force)

        return {
            "images": torch.stack(image_tensors, dim=0),
            "history_pose_deltas": torch.tensor(np.stack(history_pose_deltas, axis=0), dtype=torch.float32),
            "force_sequence": torch.tensor(np.stack(force_sequence, axis=0), dtype=torch.float32),
            "target_pose_delta": torch.tensor(sample.target_pose_delta, dtype=torch.float32),
        }


class SingleFramePolicyDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_transform,
        window_size: int = 8,
        pose_representation: str = "pose7_quaternion",
    ) -> None:
        self.samples = load_manifest_samples(
            manifest_path=manifest_path,
            window_size=window_size,
            pose_representation=pose_representation,
            use_force=False,
            force_dim=1,
        )
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = Image.merge("RGB", (image, image, image))
        return self.image_transform(image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        current_frame = sample.frame_records[-1]
        return {
            "image": self._load_rgb_image(current_frame.image_path),
            "target_pose_delta": torch.tensor(sample.target_pose_delta, dtype=torch.float32),
        }
