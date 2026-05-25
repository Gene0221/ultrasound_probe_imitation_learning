from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .pose import normalize_pose_delta_vector, pose_representation_dim, pose_sequence_pad_value


@dataclass
class TemporalPolicyState:
    window_size: int
    pose_representation: str
    use_force: bool
    force_dim: int

    def __post_init__(self) -> None:
        self.image_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.pose_delta_buffer: deque[np.ndarray] = deque(maxlen=self.window_size - 1)
        self.force_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def append_observation(self, image_array: np.ndarray, force_vector: Iterable[float] | None = None) -> None:
        self.image_buffer.append(image_array)
        if self.use_force:
            if force_vector is None:
                self.force_buffer.append(np.zeros(self.force_dim, dtype=np.float32))
            else:
                self.force_buffer.append(np.asarray(force_vector, dtype=np.float32).reshape(self.force_dim))

    def append_executed_pose_delta(self, pose_delta_vector: Iterable[float]) -> None:
        pose_feature = normalize_pose_delta_vector(
            pose_delta_vector,
            representation=self.pose_representation,
        )
        self.pose_delta_buffer.append(pose_feature)

    def is_ready(self) -> bool:
        return len(self.image_buffer) == self.window_size and (
            not self.use_force or len(self.force_buffer) == self.window_size
        )


def build_temporal_inputs(
    state: TemporalPolicyState,
    image_transform,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not state.is_ready():
        raise RuntimeError("State buffer is not full yet.")

    image_tensor = torch.stack(
        [preprocess_grayscale_image(image, image_transform) for image in state.image_buffer],
        dim=0,
    ).unsqueeze(0).to(device)

    pose_history = list(state.pose_delta_buffer)
    pose_pad = pose_sequence_pad_value(state.pose_representation)
    while len(pose_history) < state.window_size - 1:
        pose_history.insert(0, pose_pad.copy())
    pose_history.append(pose_pad.copy())
    history_pose_tensor = torch.tensor(np.stack(pose_history, axis=0), dtype=torch.float32).unsqueeze(0).to(device)

    if state.use_force:
        force_history = list(state.force_buffer)
    else:
        force_history = [np.zeros(state.force_dim, dtype=np.float32) for _ in range(state.window_size)]
    force_tensor = torch.tensor(np.stack(force_history, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    return image_tensor, history_pose_tensor, force_tensor


def preprocess_grayscale_image(image_array: np.ndarray, image_transform) -> torch.Tensor:
    image = Image.fromarray(image_array.astype(np.uint8), mode="L")
    image = Image.merge("RGB", (image, image, image))
    return image_transform(image)
