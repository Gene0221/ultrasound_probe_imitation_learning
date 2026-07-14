from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments
    def tqdm(iterable, **_):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diffusion_ultrasound.dataset import DiffusionActionChunkDataset  # noqa: E402
from diffusion_ultrasound.model import LinearNoiseScheduler, ResNet18ConditionedDenoiser  # noqa: E402
from diffusion_ultrasound.train_utils import save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18-conditioned action-chunk diffusion model.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "train.yaml"))
    return parser.parse_args()


def build_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def run_epoch(model, scheduler, loader, optimizer, device, include_force: bool, *, epoch: int, mode: str) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_count = 0
    progress = tqdm(loader, desc=f"{mode} epoch {epoch}", unit="batch", leave=False)
    for batch in progress:
        image = batch["image"].to(device)
        action = batch["action"].to(device)
        force = batch.get("force")
        if force is not None:
            force = force.to(device)
        timesteps = torch.randint(0, scheduler.num_steps, (action.shape[0],), device=device)
        noise = torch.randn_like(action)
        noisy_action = scheduler.add_noise(action, noise, timesteps)
        with torch.set_grad_enabled(optimizer is not None):
            predicted_noise = model(noisy_action, image, timesteps, force if include_force else None)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * image.shape[0]
        total_count += image.shape[0]
        avg_loss = total_loss / max(total_count, 1)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{float(loss.item()):.6f}", avg=f"{avg_loss:.6f}")
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config["training"].get("seed", 42)))
    version = str(config["dataset"].get("version", "without_force"))
    include_force = version == "with_force"
    image_size = int(config["dataset"].get("image_size", 224))
    transform = build_transform(image_size)

    train_dataset = DiffusionActionChunkDataset(config["dataset"]["dataset_root"], "train", version, transform=transform)
    val_dataset = DiffusionActionChunkDataset(config["dataset"]["dataset_root"], "val", version, transform=transform)
    test_dataset = DiffusionActionChunkDataset(config["dataset"]["dataset_root"], "test", version, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"].get("num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"].get("num_workers", 0)))
    test_loader = DataLoader(test_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"].get("num_workers", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18ConditionedDenoiser(
        action_horizon=int(config["model"].get("action_horizon", config["dataset"].get("action_horizon", 20))),
        action_dim=int(config["model"].get("action_dim", 7)),
        force_dim=int(config["model"].get("force_dim", 1 if include_force else 0)) if include_force else 0,
        hidden_dim=int(config["model"].get("hidden_dim", 1024)),
        time_embed_dim=int(config["model"].get("time_embed_dim", 128)),
        pretrained=bool(config["model"].get("pretrained_resnet18", True)),
        freeze_encoder=bool(config["model"].get("freeze_encoder", False)),
    ).to(device)
    scheduler = LinearNoiseScheduler(
        num_steps=int(config["diffusion"].get("num_steps", 100)),
        beta_start=float(config["diffusion"].get("beta_start", 0.0001)),
        beta_end=float(config["diffusion"].get("beta_end", 0.02)),
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0.0)))

    output_dir = Path(str(config["training"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = run_epoch(model, scheduler, train_loader, optimizer, device, include_force, epoch=epoch, mode="train")
        val_loss = run_epoch(model, scheduler, val_loader, None, device, include_force, epoch=epoch, mode="val")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": config, "history": history}, output_dir / "best.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = run_epoch(model, scheduler, test_loader, None, device, include_force, epoch=0, mode="test")
    print(f"test_loss={test_loss:.6f}")
    save_json(
        output_dir / "summary.json",
        {"best_val_loss": best_val, "test_loss": test_loss, "history": history, "config": config},
    )


if __name__ == "__main__":
    main()

