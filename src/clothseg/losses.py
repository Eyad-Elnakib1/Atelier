"""Training loss: cross-entropy + Lovasz-Softmax.

Cross-entropy gives stable per-pixel gradients but is dominated by large classes.
Lovasz-Softmax (Berman et al., CVPR 2018, https://arxiv.org/abs/1705.08790) is a
differentiable surrogate of the per-class IoU, so a belt counts as much as a pair
of trousers. The 0.5/0.5 mix is the segmentation loss of SCHP, the strongest
published method on ATR.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .labels import IGNORE_INDEX


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard loss w.r.t. sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(probs: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Lovasz-Softmax over the whole batch, averaged over the classes present in it.

    Args:
        probs: (B, C, H, W) class probabilities (float32).
        target: (B, H, W) integer labels.
    """
    c = probs.shape[1]
    probs = probs.permute(0, 2, 3, 1).reshape(-1, c)
    target = target.reshape(-1)
    valid = target != ignore_index
    probs, target = probs[valid], target[valid]
    if target.numel() == 0:
        return probs.sum() * 0.0

    losses = []
    for cls in torch.unique(target).tolist():  # classes absent from the batch are skipped
        fg = (target == cls).float()
        errors = (fg - probs[:, cls]).abs()
        errors_sorted, order = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[order])))
    return torch.stack(losses).mean()


class SegLoss(nn.Module):
    """``ce_weight * CrossEntropy + lovasz_weight * LovaszSoftmax``, computed in float32."""

    def __init__(self, ce_weight: float = 0.5, lovasz_weight: float = 0.5, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ce_weight, self.lovasz_weight, self.ignore_index = ce_weight, lovasz_weight, ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        # fp16 logits are cast up: sorting and cumulative sums over millions of pixels
        # lose too much precision in half floats.
        logits = logits.float()
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        parts = {"ce": float(ce.detach())}
        loss = self.ce_weight * ce
        if self.lovasz_weight > 0:
            lov = lovasz_softmax(logits.softmax(dim=1), target, self.ignore_index)
            parts["lovasz"] = float(lov.detach())
            loss = loss + self.lovasz_weight * lov
        else:
            parts["lovasz"] = 0.0
        return loss, parts
