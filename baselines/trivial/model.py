"""Trivial pose-regression model: ResNet-18 backbone + pose heads."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class PoseRegressor(nn.Module):
    def __init__(self, in_channels: int = 2, embedding_dim: int = 512):
        super().__init__()
        backbone = models.resnet18(weights=None)
        # Swap first conv for n-channel input.
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.embed_head = nn.Linear(feat_dim, embedding_dim)
        self.rotation_head = nn.Linear(embedding_dim, 6)
        self.translation_head = nn.Linear(embedding_dim, 3)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        emb = self.embed_head(feat)
        rot6 = self.rotation_head(emb)
        trans = self.translation_head(emb)
        return emb, rot6, trans
