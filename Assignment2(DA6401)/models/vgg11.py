"""VGG11 backbone - implements VGG11 per the official paper (Simonyan & Zisserman, 2014).
BatchNorm and CustomDropout injected as design choices.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11(nn.Module):


    def __init__(self, num_classes: int = 1000, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        # Block 1: 1 conv, 64 filters -> 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 1 conv, 128 filters -> 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 2 convs, 256 filters -> 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 2 convs, 512 filters -> 28x28 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 5: 2 convs, 512 filters -> 14x14 -> 7x7
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # FC classifier: 3 layers as in VGG paper
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        features = {}

        f1 = self._block_with_skip(self.block1, x)
        features["block1"] = f1["pre_pool"]
        out = f1["out"]

        f2 = self._block_with_skip(self.block2, out)
        features["block2"] = f2["pre_pool"]
        out = f2["out"]

        f3 = self._block_with_skip(self.block3, out)
        features["block3"] = f3["pre_pool"]
        out = f3["out"]

        f4 = self._block_with_skip(self.block4, out)
        features["block4"] = f4["pre_pool"]
        out = f4["out"]

        f5 = self._block_with_skip(self.block5, out)
        features["block5"] = f5["pre_pool"]
        bottleneck = f5["out"]

        if return_features:
            return bottleneck, features

        return self.classifier(bottleneck)

    def _block_with_skip(self, block: nn.Sequential, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        pre_pool = x
        for layer in block:
            if isinstance(layer, nn.MaxPool2d):
                out = layer(pre_pool)
                return {"pre_pool": pre_pool, "out": out}
            pre_pool = layer(pre_pool)
        return {"pre_pool": pre_pool, "out": pre_pool}



VGG11Encoder = VGG11
