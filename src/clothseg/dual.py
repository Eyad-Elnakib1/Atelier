"""A two-encoder segmentation network: a transformer and a convolutional encoder, fused.

Motivation. In the model comparison, SegFormer (transformer encoder) and UPerNet +
ConvNeXt (convolutional encoder) reached similar scores with different designs. The
transformer relates distant parts of the image through attention; the convolutional
network is strong on local detail and texture. Parallel CNN/transformer designs with a
fusion step are published (Conformer, Peng et al. 2021; TransFuse, Zhang et al. 2021).
This network follows that idea with the two encoders we already use.

Structure::

    photo ──► MiT encoder (SegFormer) ──► 4 feature maps at 1/4, 1/8, 1/16, 1/32
          └─► ConvNeXt encoder        ──► 4 feature maps at the same scales
                     │
                     ▼  per scale: FusionBlock (project both, gate the channels, merge)
               4 fused maps
                     │
                     ▼  FusionDecoder: pyramid pooling on the coarsest map (global
                        context), top-down pathway with lateral additions, then all
                        four levels are brought to 1/4 scale, concatenated and classified

Both encoders start from their own ImageNet weights. The fusion blocks and the decoder
are new. During training each encoder also gets a small auxiliary classifier on its
1/16-scale features, so that neither branch can be ignored by the fusion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _conv_bn_relu(c_in: int, c_out: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class FusionBlock(nn.Module):
    """Joins the two encoders' feature maps at one scale.

    Each map is projected to the same width; the two are concatenated; a channel gate
    (squeeze-and-excitation style: global average, two 1x1 convolutions, sigmoid) lets the
    network weight each encoder's channels per image; a 3x3 convolution merges them.
    """

    def __init__(self, c_a: int, c_b: int, dim: int) -> None:
        super().__init__()
        self.proj_a = _conv_bn_relu(c_a, dim, 1)
        self.proj_b = _conv_bn_relu(c_b, dim, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(2 * dim, dim // 2, 1), nn.ReLU(inplace=True), nn.Conv2d(dim // 2, 2 * dim, 1), nn.Sigmoid()
        )
        self.merge = _conv_bn_relu(2 * dim, dim, 3)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if b.shape[-2:] != a.shape[-2:]:  # the two encoders can round odd sizes differently
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([self.proj_a(a), self.proj_b(b)], dim=1)
        return self.merge(x * self.gate(x))


class PyramidPooling(nn.Module):
    """Global context for the coarsest map: average pooling at several grid sizes (as in PSPNet/UPerNet)."""

    def __init__(self, dim: int, bins: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        # No BatchNorm here: a 1x1 pooled map with a small batch has too few values to normalise.
        self.stages = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(dim, dim // 4, 1), nn.ReLU(inplace=True)) for b in bins
        )
        self.merge = _conv_bn_relu(dim + len(bins) * (dim // 4), dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False) for stage in self.stages]
        return self.merge(torch.cat([x] + pooled, dim=1))


class FusionDecoder(nn.Module):
    """Decoder for the four fused maps.

    Pyramid pooling on the coarsest level, a top-down pathway in which each finer level
    receives the upsampled coarser one, then all levels at 1/4 scale are concatenated and
    classified. The first two ideas come from UPerNet, the last from SegFormer.
    """

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.context = PyramidPooling(dim)
        self.smooth = nn.ModuleList(_conv_bn_relu(dim, dim, 3) for _ in range(3))
        self.merge = _conv_bn_relu(4 * dim, dim, 3)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(dim, num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        levels = [None, None, None, self.context(feats[3])]
        for i in (2, 1, 0):  # top-down: coarse information flows into the finer levels
            up = F.interpolate(levels[i + 1], size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            levels[i] = self.smooth[i](feats[i] + up)
        size = levels[0].shape[-2:]
        stacked = [levels[0]] + [F.interpolate(x, size=size, mode="bilinear", align_corners=False) for x in levels[1:]]
        return self.classifier(self.dropout(self.merge(torch.cat(stacked, dim=1))))


class DualEncoderNet(nn.Module):
    """MiT (SegFormer) encoder + ConvNeXt encoder -> per-scale fusion -> FusionDecoder.

    ``forward`` returns class scores at 1/4 of the input size. In training mode the
    auxiliary scores of the two encoders are kept for :meth:`aux_loss`.
    """

    AUX_WEIGHT = 0.4  # weight of each encoder's auxiliary loss

    def __init__(self, mit: nn.Module, mit_channels: list[int], convnext: nn.Module, convnext_channels: list[int],
                 num_classes: int, dim: int = 256) -> None:
        super().__init__()
        self.mit, self.convnext = mit, convnext
        self.fusion = nn.ModuleList(FusionBlock(a, b, dim) for a, b in zip(mit_channels, convnext_channels))
        self.decoder = FusionDecoder(dim, num_classes)
        self.aux_mit = nn.Conv2d(mit_channels[2], num_classes, 1)
        self.aux_convnext = nn.Conv2d(convnext_channels[2], num_classes, 1)
        self._aux: tuple[torch.Tensor, torch.Tensor] | None = None

    def encoders(self) -> nn.ModuleList:
        return nn.ModuleList([self.mit, self.convnext])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mit(x, output_hidden_states=True).hidden_states
        b = self.convnext(x)
        if len(a) != 4 or len(b) != 4 or a[0].ndim != 4:
            raise RuntimeError("The encoders did not return four feature maps each; the installed library versions are not supported.")
        if self.training:
            self._aux = (self.aux_mit(a[2]), self.aux_convnext(b[2]))
        return self.decoder([fuse(fa, fb) for fuse, fa, fb in zip(self.fusion, a, b)])

    def aux_loss(self, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]] | None:
        if self._aux is None:
            return None
        outputs, self._aux = self._aux, None
        losses = [
            F.cross_entropy(F.interpolate(o.float(), size=target.shape[-2:], mode="bilinear", align_corners=False), target, ignore_index=255)
            for o in outputs
        ]
        total = self.AUX_WEIGHT * (losses[0] + losses[1])
        return total, {"aux_mit": float(losses[0].detach()), "aux_convnext": float(losses[1].detach())}


def build_dual(mit_size: str, num_classes: int, pretrained: bool) -> DualEncoderNet:
    """Build the two-encoder network with a MiT encoder of the given size (``b0`` ... ``b5``) and ConvNeXt-Tiny."""
    import timm
    from transformers import SegformerConfig, SegformerModel
    from transformers.utils import logging as hf_logging

    repo = f"nvidia/mit-{mit_size}"
    if pretrained:
        # The checkpoint also holds an ImageNet classifier that this encoder-only model does not
        # use; transformers reports that at length, so the report is silenced.
        previous = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            mit = SegformerModel.from_pretrained(repo)
        finally:
            hf_logging.set_verbosity(previous)
    else:
        mit = SegformerModel(SegformerConfig.from_pretrained(repo))
    convnext = timm.create_model("convnext_tiny", pretrained=pretrained, features_only=True)
    return DualEncoderNet(mit, list(mit.config.hidden_sizes), convnext, list(convnext.feature_info.channels()), num_classes)
