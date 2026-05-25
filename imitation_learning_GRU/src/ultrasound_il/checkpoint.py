from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, Path(path))
