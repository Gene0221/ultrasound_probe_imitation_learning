from __future__ import annotations

import torch
from torch import nn

from .backbones import ResNet18FeatureExtractor


class PoseMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pose_deltas: torch.Tensor) -> torch.Tensor:
        return self.net(pose_deltas)


class ForceMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, force_sequence: torch.Tensor) -> torch.Tensor:
        return self.net(force_sequence)


class TemporalPolicyNet(nn.Module):
    def __init__(
        self,
        pose_output_dim: int,
        pose_input_dim: int,
        use_force: bool,
        force_dim: int,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        pose_feature_dim: int = 64,
        force_feature_dim: int = 32,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pose_output_dim = pose_output_dim
        self.use_force = use_force

        self.image_encoder = ResNet18FeatureExtractor(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )
        self.pose_encoder = PoseMLPEncoder(
            input_dim=pose_input_dim,
            output_dim=pose_feature_dim,
            dropout=dropout,
        )
        self.force_encoder = (
            ForceMLPEncoder(input_dim=force_dim, output_dim=force_feature_dim, dropout=dropout)
            if use_force
            else None
        )

        lstm_input_dim = self.image_encoder.output_dim + pose_feature_dim + (force_feature_dim if use_force else 0)
        effective_lstm_dropout = lstm_dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=effective_lstm_dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, pose_output_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        history_pose_deltas: torch.Tensor,
        force_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = images.shape[:2]
        image_features = self.image_encoder(images.view(batch_size * sequence_length, *images.shape[2:]))
        image_features = image_features.view(batch_size, sequence_length, -1)

        pose_features = self.pose_encoder(history_pose_deltas)
        feature_parts = [image_features, pose_features]

        if self.use_force:
            if force_sequence is None:
                raise ValueError("force_sequence is required when use_force=True.")
            force_features = self.force_encoder(force_sequence)
            feature_parts.append(force_features)

        sequence_features = torch.cat(feature_parts, dim=-1)
        _, (hidden, _) = self.lstm(sequence_features)
        final_hidden = hidden[-1]
        return self.regressor(final_hidden)
