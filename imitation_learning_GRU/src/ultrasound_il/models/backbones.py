from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 512

        if freeze_backbone:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return features.flatten(1)
