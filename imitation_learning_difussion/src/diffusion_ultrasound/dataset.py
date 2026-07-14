from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from PIL import Image
import torch
from torch.utils.data import Dataset


class DiffusionActionChunkDataset(Dataset):
    def __init__(self, dataset_root: Union[str, Path], split: str, version: str, transform=None) -> None:
        self.root = Path(dataset_root).resolve() / version / split
        self.transform = transform
        samples_path = self.root / "samples.json"
        self.samples: list[dict[str, Any]] = json.loads(samples_path.read_text(encoding="utf-8"))
        if not self.samples:
            raise FileNotFoundError(f"No samples found in {samples_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(self.root / sample["image"]).convert("RGB")
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        action = torch.tensor(sample["action_chunk"], dtype=torch.float32)
        item = {"image": image_tensor, "action": action}
        if "force" in sample:
            item["force"] = torch.tensor([float(sample["force"])], dtype=torch.float32)
        return item
