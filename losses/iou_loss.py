"""Custom IoU loss for bounding box regression."""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Computes 1 - IoU between predicted and target boxes.
    IoU is always in [0, 1], so this loss is always in [0, 1].

    Boxes must be in (x_center, y_center, width, height) format.
    Works with any coordinate space (pixel or normalized) since
    IoU is scale-invariant.

    Supports reduction types: 'mean' (default), 'sum', 'none'.

    Args:
        eps: small constant for numerical stability.
        reduction: 'mean' (default), 'sum', or 'none'.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"reduction must be one of 'none', 'mean', 'sum', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred_boxes:   [B, 4] floats in (x_center, y_center, width, height).
            target_boxes: [B, 4] floats in (x_center, y_center, width, height).

        Returns:
            Scalar loss in [0, 1] (or [B] tensor if reduction='none').
        """
        # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection area
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union area
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area  = (tgt_x2  - tgt_x1).clamp(min=0)  * (tgt_y2  - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area

        # IoU in [0,1] -> loss in [0,1]
        iou  = inter_area / (union_area + self.eps)
        loss = 1.0 - iou  # shape [B], always in [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"
