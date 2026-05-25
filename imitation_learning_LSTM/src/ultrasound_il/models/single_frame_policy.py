from __future__ import annotations

import torch
from torch import nn

from .backbones import ResNet18FeatureExtractor


class SingleFramePolicyNet(nn.Module):
    def __init__(
        self,
        pose_output_dim: int,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_encoder = ResNet18FeatureExtractor(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, pose_output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(image)
        return self.regressor(features)
