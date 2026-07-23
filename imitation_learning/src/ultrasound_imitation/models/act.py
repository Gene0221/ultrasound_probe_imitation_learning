from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class ACTPolicy(nn.Module):
    def __init__(
        self,
        *,
        action_horizon: int,
        action_dim: int = 7,
        force_dim: int = 0,
        hidden_dim: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)
        feature_dim = int(encoder.fc.in_features)
        self.image_encoder = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        if freeze_encoder:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.force_dim = force_dim
        self.feature_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.image_projection = nn.Linear(feature_dim, hidden_dim)
        self.image_position = nn.Parameter(torch.zeros(1, 49, hidden_dim))
        self.action_queries = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)
        self.force_projection = nn.Linear(force_dim, hidden_dim) if force_dim else None
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, image: torch.Tensor, force: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.image_encoder(image)
        features = self.feature_pool(features)
        features = features.flatten(2).transpose(1, 2)
        memory = self.image_projection(features) + self.image_position
        if self.force_dim:
            if force is None:
                raise ValueError("Model was configured with force_dim > 0 but no force tensor was provided.")
            force_token = self.force_projection(force).unsqueeze(1)
            memory = torch.cat([memory, force_token], dim=1)
        queries = self.action_queries.expand(image.shape[0], -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        decoded = self.output_norm(decoded)
        return self.action_head(decoded)


ResNet18ActionChunkPolicy = ACTPolicy
