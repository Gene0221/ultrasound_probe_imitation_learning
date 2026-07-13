from __future__ import annotations

import torch
from torch import nn


class ResNet18ActionChunkPolicy(nn.Module):
    def __init__(
        self,
        *,
        action_horizon: int,
        action_dim: int = 7,
        force_dim: int = 0,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)
        feature_dim = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
        if freeze_encoder:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        self.encoder = encoder
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.force_dim = force_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim + force_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_horizon * action_dim),
        )

    def forward(self, image: torch.Tensor, force: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(image)
        if self.force_dim:
            if force is None:
                raise ValueError("Model was configured with force_dim > 0 but no force tensor was provided.")
            features = torch.cat([features, force], dim=-1)
        output = self.head(features)
        return output.view(image.shape[0], self.action_horizon, self.action_dim)

