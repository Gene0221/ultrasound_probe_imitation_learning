from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding


class ResNet18ConditionedDenoiser(nn.Module):
    def __init__(
        self,
        *,
        action_horizon: int,
        action_dim: int = 7,
        force_dim: int = 0,
        hidden_dim: int = 1024,
        time_embed_dim: int = 128,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)
        image_feature_dim = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
        if freeze_encoder:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        self.encoder = encoder
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.force_dim = force_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        flat_action_dim = action_horizon * action_dim
        self.network = nn.Sequential(
            nn.Linear(flat_action_dim + image_feature_dim + force_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

    def forward(self, noisy_action: torch.Tensor, image: torch.Tensor, timesteps: torch.Tensor, force: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.encoder(image)
        time_features = self.time_embedding(timesteps)
        flat_action = noisy_action.flatten(start_dim=1)
        inputs = [flat_action, image_features, time_features]
        if self.force_dim:
            if force is None:
                raise ValueError("Model was configured with force_dim > 0 but no force tensor was provided.")
            inputs.append(force)
        output = self.network(torch.cat(inputs, dim=-1))
        return output.view(noisy_action.shape)


class LinearNoiseScheduler:
    def __init__(self, num_steps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise

