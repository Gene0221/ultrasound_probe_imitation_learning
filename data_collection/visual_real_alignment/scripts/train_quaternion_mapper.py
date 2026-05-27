from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_PATH = WORKSPACE_ROOT / "output" / "paired_quaternion_dataset.pt"
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "model" / "quaternion_mapper_mlp.pt"
DEFAULT_METRICS_PATH = WORKSPACE_ROOT / "model" / "quaternion_mapper_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a quaternion mapper MLP.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the paired dataset .pt file.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved model checkpoint.")
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH, help="Path to the metrics JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width for the MLP.")
    parser.add_argument("--depth", type=int, default=3, help="Number of linear layers including output.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QuaternionPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.inputs = payload["visual_quat_xyzw"].to(dtype=torch.float32)
        self.targets = payload["real_quat_xyzw"].to(dtype=torch.float32)
        if self.inputs.ndim != 2 or self.inputs.shape[1] != 4:
            raise ValueError("visual_quat_xyzw must have shape [N, 4].")
        if self.targets.shape != self.inputs.shape:
            raise ValueError("real_quat_xyzw must match visual_quat_xyzw.")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def normalized_quaternion(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    return quaternion_xyzw / quaternion_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quaternion_regression_loss(prediction_xyzw: torch.Tensor, target_xyzw: torch.Tensor) -> torch.Tensor:
    prediction_xyzw = normalized_quaternion(prediction_xyzw)
    target_xyzw = normalized_quaternion(target_xyzw)
    direct = torch.mean((prediction_xyzw - target_xyzw) ** 2, dim=-1)
    flipped = torch.mean((prediction_xyzw + target_xyzw) ** 2, dim=-1)
    return torch.minimum(direct, flipped).mean()


def quaternion_angle_error_deg(prediction_xyzw: torch.Tensor, target_xyzw: torch.Tensor) -> torch.Tensor:
    prediction_xyzw = normalized_quaternion(prediction_xyzw)
    target_xyzw = normalized_quaternion(target_xyzw)
    dot_values = torch.abs(torch.sum(prediction_xyzw * target_xyzw, dim=-1)).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.arccos(dot_values))


class QuaternionMapperMLP(nn.Module):
    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")

        layers: list[nn.Module] = []
        input_dim = 4
        for _ in range(depth - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs_xyzw: torch.Tensor) -> torch.Tensor:
        raw_output = self.network(inputs_xyzw)
        return normalized_quaternion(raw_output)


@dataclass
class SplitResult:
    train: Subset
    val: Subset
    test: Subset


def split_dataset(
    dataset: QuaternionPairDataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitResult:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")
    if len(dataset) < 3:
        raise ValueError("At least 3 samples are required to build train/val/test splits.")

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_count = max(1, int(len(indices) * train_ratio))
    val_count = max(1, int(len(indices) * val_ratio))
    remaining = len(indices) - train_count - val_count
    if remaining <= 0:
        val_count = max(1, val_count - 1)
        remaining = len(indices) - train_count - val_count
    test_count = remaining
    if test_count <= 0:
        raise ValueError("Split ratios leave no samples for the test set.")

    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count : train_count + val_count + test_count]
    return SplitResult(
        train=Subset(dataset, train_indices),
        val=Subset(dataset, val_indices),
        test=Subset(dataset, test_indices),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    angle_errors: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs_xyzw, targets_xyzw in loader:
            inputs_xyzw = inputs_xyzw.to(device)
            targets_xyzw = targets_xyzw.to(device)
            predictions_xyzw = model(inputs_xyzw)
            loss = quaternion_regression_loss(predictions_xyzw, targets_xyzw)
            losses.append(float(loss.item()))
            angle_errors.append(quaternion_angle_error_deg(predictions_xyzw, targets_xyzw).cpu())

    concatenated_errors = torch.cat(angle_errors) if angle_errors else torch.zeros(0, dtype=torch.float32)
    return {
        "loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "angle_error_deg_mean": float(concatenated_errors.mean().item()) if concatenated_errors.numel() else 0.0,
        "angle_error_deg_median": float(concatenated_errors.median().item()) if concatenated_errors.numel() else 0.0,
        "angle_error_deg_max": float(concatenated_errors.max().item()) if concatenated_errors.numel() else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = QuaternionPairDataset(args.dataset)
    splits = split_dataset(
        dataset=dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(splits.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)

    model = QuaternionMapperMLP(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for inputs_xyzw, targets_xyzw in train_loader:
            inputs_xyzw = inputs_xyzw.to(device)
            targets_xyzw = targets_xyzw.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions_xyzw = model(inputs_xyzw)
            loss = quaternion_regression_loss(predictions_xyzw, targets_xyzw)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)
        val_metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_angle_error_deg_mean": float(val_metrics["angle_error_deg_mean"]),
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"[EPOCH {epoch:03d}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_angle_mean_deg={val_metrics['angle_error_deg_mean']:.6f}"
            )

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state_dict)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    checkpoint = {
        "model_state_dict": best_state_dict,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
        },
        "train_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "history": history,
    }

    metrics_report = {
        "dataset_path": str(args.dataset),
        "device": str(device),
        "sample_counts": {
            "total": len(dataset),
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "metrics": checkpoint["metrics"],
    }

    ensure_parent(args.model_out)
    ensure_parent(args.metrics_out)
    torch.save(checkpoint, args.model_out)
    args.metrics_out.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")

    print(f"[DONE] Saved model checkpoint to: {args.model_out}")
    print(f"[DONE] Saved metrics to: {args.metrics_out}")
    print(
        "[TEST] "
        f"loss={test_metrics['loss']:.6f} "
        f"mean_deg={test_metrics['angle_error_deg_mean']:.6f} "
        f"median_deg={test_metrics['angle_error_deg_median']:.6f} "
        f"max_deg={test_metrics['angle_error_deg_max']:.6f}"
    )


if __name__ == "__main__":
    main()
