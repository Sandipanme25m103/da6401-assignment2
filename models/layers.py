"""Reusable custom layers
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer implemented from scratch (without nn.Dropout).

    During training, each neuron activation is independently zeroed with
    probability `p`, and the remaining activations are scaled by 1/(1-p)
    (inverted dropout) to preserve expected values at inference time.

    During evaluation, the layer is a no-op (identity).

    Args:
        p: probability of an element being zeroed. Must be in [0, 1).
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to input tensor.

        Args:
            x: input tensor of any shape.

        Returns:
            Tensor of same shape as x with dropout applied during training,
            or x unchanged during evaluation.
        """
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
