"""Arquitecturas: CNN desde cero y ResNet-18 (transfer learning / fine tuning)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

SCHEMES = {
    "scratch": "Desde cero (CNN propia)",
    "transfer": "Transfer learning (ResNet-18 congelada)",
    "finetune": "Fine tuning (ResNet-18, layer3-4 descongeladas)",
}


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    def forward(self, x):
        return (x - self.mean) / self.std


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    """CNN de 4 bloques conv-BN-ReLU-MaxPool (32-64-128-128 filtros) + global average pooling."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.norm = Normalize()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(self.norm(x)))


class ResNetClassifier(nn.Module):
    def __init__(self, n_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.norm = Normalize()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.net.fc.in_features, n_classes))

    def forward(self, x):
        return self.net(self.norm(x))

    def freeze_backbone(self):
        for name, p in self.net.named_parameters():
            p.requires_grad = name.startswith("fc.")

    def unfreeze_top(self):
        for name, p in self.net.named_parameters():
            p.requires_grad = name.startswith(("layer3.", "layer4.", "fc."))


def build_model(scheme: str, pretrained: bool = True) -> nn.Module:
    if scheme == "scratch":
        return ScratchCNN()
    model = ResNetClassifier(pretrained=pretrained)
    if scheme == "transfer":
        model.freeze_backbone()
    else:
        model.unfreeze_top()
    return model
