"""Reusable custom layers
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):


    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or self.p == 0.0:
            # At inference time, return input unchanged
            return x

        # Sample a binary mask: 1 with prob (1-p), 0 with prob p
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))

        # Inverted dropout: scale kept activations by 1/(1-p)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
