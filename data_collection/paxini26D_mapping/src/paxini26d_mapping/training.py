from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import ensure_dir, save_json, utc_now_iso


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class SplitData:
    features: torch.Tensor
    targets: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(num_items: int, train_ratio: float, val_ratio: float) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(num_items))
    random.shuffle(indices)
    train_end = int(num_items * train_ratio)
    val_end = train_end + int(num_items * val_ratio)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    if not train_indices or not val_indices or not test_indices:
        raise RuntimeError("Dataset split is empty. Collect more sessions or adjust split ratios.")
    return train_indices, val_indices, test_indices


def index_tensor(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return tensor[torch.tensor(indices, dtype=torch.long)]


def normalize_from_train(
    train_features: torch.Tensor,
    other_splits: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0)
    feature_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)
    normalized_train = (train_features - feature_mean) / feature_std
    normalized_others = [(split - feature_mean) / feature_std for split in other_splits]
    return normalized_train, normalized_others, feature_mean, feature_std


def evaluate(model: nn.Module, split: SplitData, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        predictions = model(split.features.to(device)).cpu()
        targets = split.targets.cpu()
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def train_model(
    dataset_payload: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    train_cfg = config["training"]
    set_seed(int(train_cfg["seed"]))

    features: torch.Tensor = dataset_payload["features"]
    targets: torch.Tensor = dataset_payload["targets"]
    train_indices, val_indices, test_indices = split_indices(
        num_items=features.shape[0],
        train_ratio=float(train_cfg["train_ratio"]),
        val_ratio=float(train_cfg["val_ratio"]),
    )

    raw_train_features = index_tensor(features, train_indices)
    raw_val_features = index_tensor(features, val_indices)
    raw_test_features = index_tensor(features, test_indices)
    train_features, [val_features, test_features], feature_mean, feature_std = normalize_from_train(
        raw_train_features,
        [raw_val_features, raw_test_features],
    )

    train_split = SplitData(train_features, index_tensor(targets, train_indices))
    val_split = SplitData(val_features, index_tensor(targets, val_indices))
    test_split = SplitData(test_features, index_tensor(targets, test_indices))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(
        input_dim=int(features.shape[1]),
        output_dim=int(targets.shape[1]),
        hidden_dims=[int(dim) for dim in train_cfg["hidden_dims"]],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(train_split.features, train_split.targets),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
    )

    best_val_mse = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    history: list[dict[str, float]] = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.shape[0]

        train_mse = epoch_loss / len(loader.dataset)
        val_metrics = evaluate(model, val_split, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": float(train_mse),
                "val_mse": float(val_metrics["mse"]),
                "val_mae": float(val_metrics["mae"]),
            }
        )
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")

    model.load_state_dict(best_state)
    final_train_metrics = evaluate(model, train_split, device)
    final_val_metrics = evaluate(model, val_split, device)
    final_test_metrics = evaluate(model, test_split, device)

    run_dir = ensure_dir(output_root / f"run_{utc_now_iso().replace(':', '-').replace('+00:00', 'Z')}")
    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_names": dataset_payload["feature_names"],
            "target_names": dataset_payload["target_names"],
            "config": config,
        },
        checkpoint_path,
    )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_samples": int(features.shape[0]),
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "num_test": len(test_indices),
        "device": str(device),
        "train_metrics": final_train_metrics,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "history_tail": history[-10:],
    }
    save_json(run_dir / "summary.json", summary)
    return summary
