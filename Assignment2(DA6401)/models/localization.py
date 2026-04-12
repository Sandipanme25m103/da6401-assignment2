"""Localization model - predicts bounding boxes in pixel space [0, 224]."""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class VGG11Localizer(nn.Module):


    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, image_size: int = 224):
        super().__init__()
        self.image_size = image_size

        # Encoder — also accessible as `encoder` via property for backward compat
        self.backbone = VGG11(num_classes=1000, in_channels=in_channels, dropout_p=dropout_p)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid(),  # -> [0, 1]; scaled to pixel space in forward
        )

    @property
    def encoder(self):
        return self.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        bottleneck, _ = self.backbone(x, return_features=True)
        # Scale Sigmoid [0,1] output to pixel space [0, 224]
        return self.regression_head(bottleneck) * self.image_size
