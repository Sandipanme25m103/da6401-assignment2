"""Localization model - predicts bounding boxes in normalized [0, 1] space."""

import torch
import torch.nn as nn

from .vgg11 import VGG11
from .layers import CustomDropout


class LocalizationHead(nn.Module):
    """Regression head: bottleneck -> (x_center, y_center, width, height) normalized to [0, 1].

    Dataset returns bbox in normalized [0, 1] coordinates, so the head outputs
    values in the same space via Sigmoid activation (no pixel-space scaling).

    Args:
        in_features: channels from encoder bottleneck (default 512).
        dropout_p: dropout probability (default 0.5).
    """

    def __init__(self, in_features: int = 512, dropout_p: float = 0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        # Sigmoid maps output to [0, 1], matching normalized dataset bbox format
        self.regressor = nn.Sequential(
            nn.Linear(in_features * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        # Output in [0, 1] normalized space — matches dataset bbox labels
        return self.regressor(x)


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Output: [x_center, y_center, width, height] normalized to [0, 1].
    Matches the dataset which returns bboxes in normalized coordinates.
    Input image size is fixed at 224x224 per VGG paper.

    Args:
        in_channels: number of input image channels (default 3).
        dropout_p: dropout probability (default 0.5).
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11(num_classes=1000, in_channels=in_channels, dropout_p=dropout_p)
        self.head = LocalizationHead(in_features=512, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, in_channels, 224, 224] normalized input image.

        Returns:
            [B, 4] bounding box (x_center, y_center, width, height) normalized to [0, 1].
        """
        bottleneck, _ = self.encoder(x, return_features=True)
        return self.head(bottleneck)
