"""Segmentation metrics computed from a confusion matrix."""
from __future__ import annotations

import torch

from .labels import IGNORE_INDEX, class_names, garment_ids


class ConfusionMatrix:
    """Accumulates a (true, predicted) pixel-count matrix on the GPU."""

    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        valid = target != IGNORE_INDEX
        idx = target[valid].to(torch.int64) * self.num_classes + pred[valid].to(torch.int64)
        self.matrix += torch.bincount(idx, minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Return overall and per-class metrics.

        A class that never occurs in the ground truth and is never predicted has no
        defined IoU; it is reported as ``nan`` and left out of the means.
        """
        m = self.matrix.double()
        tp = m.diag()
        gt, pred = m.sum(1), m.sum(0)
        union = gt + pred - tp
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.full_like(tp, float("nan")))
        dice = torch.where(gt + pred > 0, 2 * tp / (gt + pred).clamp(min=1), torch.full_like(tp, float("nan")))
        acc = torch.where(gt > 0, tp / gt.clamp(min=1), torch.full_like(tp, float("nan")))
        g = garment_ids(self.num_classes)
        # Binary "any garment vs everything else": what a cut-out user sees.
        is_g = torch.zeros(self.num_classes, dtype=torch.bool, device=m.device)
        is_g[g] = True
        g_tp = m[is_g][:, is_g].sum()
        g_union = m[is_g].sum() + m[:, is_g].sum() - g_tp
        names = class_names(self.num_classes)
        return {
            "miou": float(iou.nanmean()),
            "garment_miou": float(iou[g].nanmean()),
            "clothing_binary_iou": float(g_tp / g_union.clamp(min=1)),
            "pixel_acc": float(tp.sum() / m.sum().clamp(min=1)),
            "mean_class_acc": float(acc.nanmean()),
            "mean_dice": float(dice.nanmean()),
            "iou": {n: float(v) for n, v in zip(names, iou)},
            "dice": {n: float(v) for n, v in zip(names, dice)},
        }
