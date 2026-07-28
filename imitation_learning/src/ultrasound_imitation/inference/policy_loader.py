from __future__ import annotations

from typing import Any

import torch

from ultrasound_imitation.models import ACTPolicy, ResNet18ConditionedDenoiser
from ultrasound_imitation.paths import resolve_path


def build_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def active_policy_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    policy_cfg = config["policy"]
    policy_type = str(policy_cfg.get("type", "act")).lower()
    typed_cfg = policy_cfg.get(policy_type)
    if isinstance(typed_cfg, dict):
        merged = {key: value for key, value in policy_cfg.items() if key not in {"act", "diffusion"}}
        merged.update(typed_cfg)
        return policy_type, merged
    return policy_type, policy_cfg


def active_motion_config(config: dict[str, Any], policy_type: str | None = None) -> dict[str, Any]:
    if policy_type is None:
        policy_type, _ = active_policy_config(config)
    motion_cfg = config.get("motion", {})
    if not isinstance(motion_cfg, dict):
        return {}
    typed_cfg = motion_cfg.get(policy_type)
    if isinstance(typed_cfg, dict):
        merged = {key: value for key, value in motion_cfg.items() if key not in {"act", "diffusion"}}
        merged.update(typed_cfg)
        return merged
    return motion_cfg


def merged_model_config(policy_cfg: dict[str, Any], train_config: dict[str, Any], key: str) -> dict[str, Any]:
    checkpoint_model_cfg = train_config.get("model", {}) if isinstance(train_config, dict) else {}
    infer_model_cfg = policy_cfg.get("model", policy_cfg.get(key, {}))
    if not isinstance(checkpoint_model_cfg, dict):
        checkpoint_model_cfg = {}
    if not isinstance(infer_model_cfg, dict):
        infer_model_cfg = {}
    merged = dict(checkpoint_model_cfg)
    merged.update(infer_model_cfg)
    return merged


def merged_diffusion_config(policy_cfg: dict[str, Any], train_config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_diffusion_cfg = train_config.get("diffusion", {}) if isinstance(train_config, dict) else {}
    infer_diffusion_cfg = policy_cfg.get("sampler", policy_cfg.get("diffusion", {}))
    if not isinstance(checkpoint_diffusion_cfg, dict):
        checkpoint_diffusion_cfg = {}
    if not isinstance(infer_diffusion_cfg, dict):
        infer_diffusion_cfg = {}
    merged = dict(checkpoint_diffusion_cfg)
    merged.update(infer_diffusion_cfg)
    return merged


class PolicyRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.policy_type, policy_cfg = active_policy_config(config)
        self.image_size = int(policy_cfg.get("image_size", 224))
        self.action_horizon = int(policy_cfg.get("action_horizon", 20))
        self.action_dim = int(policy_cfg.get("action_dim", 7))
        self.use_force_condition = bool(policy_cfg.get("use_force_condition", False))
        self.device = self._select_device(str(policy_cfg.get("device", "auto")))
        self.transform = build_transform(self.image_size)
        self.diffusion_config: dict[str, Any] = {}
        self.model = self._load_model(policy_cfg)
        self.model.eval()

    @staticmethod
    def _select_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def _checkpoint_path(self, policy_cfg: dict[str, Any]) -> Path:
        model_dir = resolve_path(str(policy_cfg["model_dir"]))
        checkpoint_name = str(policy_cfg.get("checkpoint_name", "best.pt"))
        return model_dir / checkpoint_name

    def _load_checkpoint(self, policy_cfg: dict[str, Any]) -> dict[str, Any]:
        checkpoint_path = self._checkpoint_path(policy_cfg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint must contain model_state_dict: {checkpoint_path}")
        return checkpoint

    def _load_model(self, policy_cfg: dict[str, Any]) -> torch.nn.Module:
        checkpoint = self._load_checkpoint(policy_cfg)
        train_config = checkpoint.get("config", {})

        if self.policy_type == "act":
            model_cfg = merged_model_config(policy_cfg, train_config, "act_model")
            model = ACTPolicy(
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                force_dim=1 if self.use_force_condition else 0,
                hidden_dim=int(model_cfg.get("hidden_dim", 512)),
                nhead=int(model_cfg.get("nhead", 8)),
                num_decoder_layers=int(model_cfg.get("num_decoder_layers", 4)),
                dim_feedforward=int(model_cfg.get("dim_feedforward", 2048)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                pretrained=bool(model_cfg.get("pretrained_resnet18", True)),
                freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
            ).to(self.device)
        elif self.policy_type == "diffusion":
            model_cfg = merged_model_config(policy_cfg, train_config, "diffusion_model")
            self.diffusion_config = merged_diffusion_config(policy_cfg, train_config)
            model = ResNet18ConditionedDenoiser(
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                force_dim=1 if self.use_force_condition else 0,
                hidden_dim=int(model_cfg.get("hidden_dim", 1024)),
                time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
                pretrained=bool(model_cfg.get("pretrained_resnet18", True)),
                freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported policy.type: {self.policy_type}")

        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @torch.inference_mode()
    def predict(self, image) -> list[list[float]]:
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        force = None
        if self.use_force_condition:
            raise NotImplementedError("Force-conditioned inference is disabled for without_force models.")

        if self.policy_type == "act":
            action = self.model(image_tensor, force)[0]
        else:
            action = self._sample_diffusion(image_tensor, force)[0]

        if action.shape != (self.action_horizon, self.action_dim):
            raise ValueError(f"Expected action shape {(self.action_horizon, self.action_dim)}, got {tuple(action.shape)}")
        return action.detach().cpu().tolist()

    def _sample_diffusion(self, image_tensor: torch.Tensor, force: torch.Tensor | None) -> torch.Tensor:
        batch_size = image_tensor.shape[0]
        num_steps = int(self.diffusion_config.get("num_steps", 100))
        beta_start = float(self.diffusion_config.get("beta_start", 0.0001))
        beta_end = float(self.diffusion_config.get("beta_end", 0.02))
        betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        action = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.device)
        for step in reversed(range(num_steps)):
            timesteps = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            predicted_noise = self.model(action, image_tensor, timesteps, force)
            beta = betas[step]
            alpha = alphas[step]
            alpha_bar = alpha_bars[step]
            action = (action - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if step > 0:
                action = action + torch.sqrt(beta) * torch.randn_like(action)
        return action
