"""Domain-adversarial model: trivial backbone + gradient-reversal + domain classifier."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class DomainAdaptedRegressor(nn.Module):
    def __init__(self, in_channels: int = 2, embedding_dim: int = 512):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.embed_head = nn.Linear(feat_dim, embedding_dim)
        self.rotation_head = nn.Linear(embedding_dim, 6)
        self.translation_head = nn.Linear(embedding_dim, 3)
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        feat = self.backbone(x)
        emb = self.embed_head(feat)
        rot6 = self.rotation_head(emb)
        trans = self.translation_head(emb)
        reversed_emb = grad_reverse(emb, alpha)
        domain_logits = self.domain_classifier(reversed_emb)
        return emb, rot6, trans, domain_logits
