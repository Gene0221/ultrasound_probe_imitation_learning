from __future__ import annotations

from typing import Iterable

import numpy as np


def normalize_pose_delta_vector(
    pose_delta: Iterable[float] | Iterable[Iterable[float]],
    representation: str = "pose7_quaternion",
) -> np.ndarray:
    if representation != "pose7_quaternion":
        raise ValueError(f"Unsupported pose representation: {representation}")

    vector = np.asarray(pose_delta, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 7:
        raise ValueError(f"Expected 7-dim pose delta [tx, ty, tz, qx, qy, qz, qw], got {vector.shape[0]}.")

    translation = vector[:3]
    quaternion = vector[3:]
    quat_norm = float(np.linalg.norm(quaternion))
    if quat_norm < 1e-8:
        quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quaternion = quaternion / quat_norm

    return np.concatenate([translation, quaternion], axis=0).astype(np.float32)


def pose_representation_dim(representation: str) -> int:
    if representation == "pose7_quaternion":
        return 7
    raise ValueError(f"Unsupported pose representation: {representation}")


def pose_sequence_pad_value(representation: str) -> np.ndarray:
    if representation == "pose7_quaternion":
        return np.zeros(7, dtype=np.float32)
    raise ValueError(f"Unsupported pose representation: {representation}")
