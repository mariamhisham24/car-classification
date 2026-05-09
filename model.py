"""
model.py
========
Transfer-learning car classifier built on top of ResNet-18.
The final fully-connected layer is replaced to match the number of car classes.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet, head replaced for `num_classes` outputs.

    Args:
        num_classes:      Number of output classes.
        freeze_backbone:  If True, all layers except the new head are frozen
                          (useful for the first few warm-up epochs).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
