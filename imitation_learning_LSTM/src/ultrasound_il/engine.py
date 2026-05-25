from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    mae: float
    translation_mae: float
    quaternion_mae: float


def _accumulate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    absolute_error = (predictions - targets).abs()
    mae = float(absolute_error.mean().item())
    translation_mae = float(absolute_error[:, :3].mean().item())
    quaternion_mae = float(absolute_error[:, 3:].mean().item())
    return mae, translation_mae, quaternion_mae


def train_one_epoch_temporal(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_force: bool,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_translation_mae = 0.0
    total_quaternion_mae = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        images = batch["images"].to(device)
        history_pose_deltas = batch["history_pose_deltas"].to(device)
        force_sequence = batch["force_sequence"].to(device) if use_force else None
        targets = batch["target_pose_delta"].to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images, history_pose_deltas, force_sequence)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        mae, translation_mae, quaternion_mae = _accumulate_metrics(predictions.detach(), targets.detach())
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_mae += mae * batch_size
        total_translation_mae += translation_mae * batch_size
        total_quaternion_mae += quaternion_mae * batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            mae=f"{total_mae / total_examples:.4f}",
        )

    return EpochMetrics(
        loss=total_loss / total_examples,
        mae=total_mae / total_examples,
        translation_mae=total_translation_mae / total_examples,
        quaternion_mae=total_quaternion_mae / total_examples,
    )


@torch.no_grad()
def evaluate_temporal(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_force: bool,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_translation_mae = 0.0
    total_quaternion_mae = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc="val", leave=False)
    for batch in progress:
        images = batch["images"].to(device)
        history_pose_deltas = batch["history_pose_deltas"].to(device)
        force_sequence = batch["force_sequence"].to(device) if use_force else None
        targets = batch["target_pose_delta"].to(device)

        predictions = model(images, history_pose_deltas, force_sequence)
        loss = criterion(predictions, targets)

        batch_size = targets.size(0)
        mae, translation_mae, quaternion_mae = _accumulate_metrics(predictions, targets)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_mae += mae * batch_size
        total_translation_mae += translation_mae * batch_size
        total_quaternion_mae += quaternion_mae * batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            mae=f"{total_mae / total_examples:.4f}",
        )

    return EpochMetrics(
        loss=total_loss / total_examples,
        mae=total_mae / total_examples,
        translation_mae=total_translation_mae / total_examples,
        quaternion_mae=total_quaternion_mae / total_examples,
    )


def train_one_epoch_single_frame(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_translation_mae = 0.0
    total_quaternion_mae = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        targets = batch["target_pose_delta"].to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        mae, translation_mae, quaternion_mae = _accumulate_metrics(predictions.detach(), targets.detach())
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_mae += mae * batch_size
        total_translation_mae += translation_mae * batch_size
        total_quaternion_mae += quaternion_mae * batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            mae=f"{total_mae / total_examples:.4f}",
        )

    return EpochMetrics(
        loss=total_loss / total_examples,
        mae=total_mae / total_examples,
        translation_mae=total_translation_mae / total_examples,
        quaternion_mae=total_quaternion_mae / total_examples,
    )


@torch.no_grad()
def evaluate_single_frame(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_translation_mae = 0.0
    total_quaternion_mae = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc="val", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        targets = batch["target_pose_delta"].to(device)

        predictions = model(images)
        loss = criterion(predictions, targets)

        batch_size = targets.size(0)
        mae, translation_mae, quaternion_mae = _accumulate_metrics(predictions, targets)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_mae += mae * batch_size
        total_translation_mae += translation_mae * batch_size
        total_quaternion_mae += quaternion_mae * batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            mae=f"{total_mae / total_examples:.4f}",
        )

    return EpochMetrics(
        loss=total_loss / total_examples,
        mae=total_mae / total_examples,
        translation_mae=total_translation_mae / total_examples,
        quaternion_mae=total_quaternion_mae / total_examples,
    )
