from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ultrasound_il.checkpoint import save_checkpoint
from ultrasound_il.config import load_config
from ultrasound_il.dataset import SingleFramePolicyDataset
from ultrasound_il.engine import evaluate_single_frame, train_one_epoch_single_frame
from ultrasound_il.models.single_frame_policy import SingleFramePolicyNet
from ultrasound_il.transforms import build_image_transform
from ultrasound_il.utils import ensure_dir, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-frame continuous ultrasound imitation baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    return parser.parse_args()


def build_loss(loss_name: str) -> nn.Module:
    normalized = loss_name.lower()
    if normalized == "smooth_l1":
        return nn.SmoothL1Loss()
    if normalized == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss type: {loss_name}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_dir = ensure_dir(PROJECT_ROOT / config["output_dir"])
    train_transform = build_image_transform(config["data"]["image_size"], is_train=True)
    val_transform = build_image_transform(config["data"]["image_size"], is_train=False)

    train_dataset = SingleFramePolicyDataset(
        manifest_path=PROJECT_ROOT / config["data"]["train_manifest"],
        image_transform=train_transform,
        window_size=config["data"]["window_size"],
        pose_representation=config["data"]["pose_representation"],
    )
    val_dataset = SingleFramePolicyDataset(
        manifest_path=PROJECT_ROOT / config["data"]["val_manifest"],
        image_transform=val_transform,
        window_size=config["data"]["window_size"],
        pose_representation=config["data"]["pose_representation"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    device = resolve_device(config["trainer"]["device"])
    model = SingleFramePolicyNet(
        pose_output_dim=config["model"]["pose_output_dim"],
        pretrained_backbone=config["model"]["pretrained_backbone"],
        freeze_backbone=config["model"]["freeze_backbone"],
        mlp_hidden_dim=config["model"]["mlp_hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = build_loss(config["trainer"]["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    best_val_loss = float("inf")
    for epoch in range(1, config["trainer"]["epochs"] + 1):
        print(f"Epoch {epoch}/{config['trainer']['epochs']}")
        train_metrics = train_one_epoch_single_frame(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_single_frame(model, val_loader, criterion, device)
        print(
            f"train_loss={train_metrics.loss:.4f} train_mae={train_metrics.mae:.4f} "
            f"train_t_mae={train_metrics.translation_mae:.4f} train_q_mae={train_metrics.quaternion_mae:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_mae={val_metrics.mae:.4f} "
            f"val_t_mae={val_metrics.translation_mae:.4f} val_q_mae={val_metrics.quaternion_mae:.4f}"
        )

        metrics_payload = {
            "train_loss": train_metrics.loss,
            "train_mae": train_metrics.mae,
            "train_translation_mae": train_metrics.translation_mae,
            "train_quaternion_mae": train_metrics.quaternion_mae,
            "val_loss": val_metrics.loss,
            "val_mae": val_metrics.mae,
            "val_translation_mae": val_metrics.translation_mae,
            "val_quaternion_mae": val_metrics.quaternion_mae,
        }

        save_checkpoint(
            output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics_payload,
            config=config,
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics_payload,
                config=config,
            )
            print(f"Saved new best checkpoint with val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
