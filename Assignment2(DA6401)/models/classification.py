"""Classification model."""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class ClassificationHead(nn.Module):
    

    def __init__(self, in_features: int = 512, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


class VGG11Classifier(nn.Module):


    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11(num_classes=1000, in_channels=in_channels, dropout_p=dropout_p)
        self.head = ClassificationHead(in_features=512, num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        bottleneck, _ = self.encoder(x, return_features=True)
        return self.head(bottleneck)
