from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from PIL import Image
import torch
from torch.utils.data import Dataset


class UltrasoundActionChunkDataset(Dataset):
    """Dataset for both ACT and diffusion action-chunk training.

    Supports both historical layouts:
    - split/samples.json with images under split/images/session_xxxx/
    - split/session_xxxx/samples.json with images under each session folder
    """

    def __init__(self, dataset_root: Union[str, Path], split: str, version: str, transform=None) -> None:
        self.root = Path(dataset_root).resolve() / version / split
        self.transform = transform
        self.samples: list[dict[str, Any]] = []

        flat_samples = self.root / "samples.json"
        if flat_samples.exists():
            records = json.loads(flat_samples.read_text(encoding="utf-8"))
            for record in records:
                item = dict(record)
                item["_base_root"] = str(self.root)
                self.samples.append(item)
        else:
            for samples_path in sorted(self.root.glob("session_*/samples.json")):
                session_root = samples_path.parent
                records = json.loads(samples_path.read_text(encoding="utf-8"))
                for record in records:
                    item = dict(record)
                    item["_base_root"] = str(session_root)
                    self.samples.append(item)

        if not self.samples:
            raise FileNotFoundError(f"No samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(Path(sample["_base_root"]) / sample["image"]).convert("RGB")
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        item = {"image": image_tensor, "action": torch.tensor(sample["action_chunk"], dtype=torch.float32)}
        if "force" in sample:
            item["force"] = torch.tensor([float(sample["force"])], dtype=torch.float32)
        return item
