"""Model factory: every candidate behind one interface.

``build_model`` returns a :class:`SegModel` whose ``forward`` gives class logits at
the input resolution, whatever the underlying library returns. The encoder is
loaded with ImageNet weights; the segmentation head starts from random values.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MODEL_NAMES = ("segformer_b0", "segformer_b2", "segformer_b3", "segformer_b4", "upernet_convnext_tiny", "deeplabv3plus_resnet101",
               "segformer_b0_edge", "segformer_b3_edge", "segformer_b0_ade", "segformer_b3_ade",
               "dual_b0_convnext_tiny", "dual_b3_convnext_tiny")


class SegModel(nn.Module):
    """Wraps a segmentation network and upsamples its logits to the input size."""

    def __init__(self, net: nn.Module, encoder: nn.Module, name: str) -> None:
        super().__init__()
        self.net = net
        self.name = name
        self._encoder_param_ids = {id(p) for p in encoder.parameters()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        logits = out.logits if hasattr(out, "logits") else out
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def aux_loss(self, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]] | None:
        """Extra training loss of the last forward pass, or ``None`` if the model has none."""
        return None

    def param_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """Four optimiser groups: encoder/head x decay/no-decay.

        The new head gets its own (higher) learning rate. Biases and normalisation
        weights (1-D parameters) are not decayed.
        """
        groups = {(part, decay): [] for part in ("encoder", "head") for decay in (True, False)}
        for p in self.parameters():
            if p.requires_grad:
                part = "encoder" if id(p) in self._encoder_param_ids else "head"
                groups[(part, p.ndim > 1)].append(p)
        return [
            {"params": params, "base_lr": encoder_lr if part == "encoder" else head_lr,
             "lr": encoder_lr if part == "encoder" else head_lr,
             "weight_decay": weight_decay if decay else 0.0, "name": f"{part}_{'decay' if decay else 'no_decay'}"}
            for (part, decay), params in groups.items() if params
        ]


class DualSegModel(SegModel):
    """Wrapper for the two-encoder network: passes its auxiliary training loss on to the training loop."""

    def aux_loss(self, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]] | None:
        return self.net.aux_loss(target)


def edge_target(label: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Binary edge map of a label map, following SCHP's ``generate_edge_tensor``.

    A pixel is an edge if it differs from its lower, right or diagonal neighbour and
    neither of the two is ``ignore_index``; the result is thickened with a 3x3 window.
    Ignored pixels stay ignored.
    """
    valid = label != ignore_index
    edge = torch.zeros_like(label, dtype=torch.bool)
    full, head, tail = slice(None), slice(0, -1), slice(1, None)
    pairs = (
        ((full, head, full), (full, tail, full)),   # pixel below
        ((full, full, head), (full, full, tail)),   # pixel to the right
        ((full, head, head), (full, tail, tail)),   # diagonal
        ((full, head, tail), (full, tail, head)),   # anti-diagonal
    )
    for a, b in pairs:
        edge[a] |= (label[a] != label[b]) & valid[a] & valid[b]
    edge = F.max_pool2d(edge.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0
    out = edge.long()
    out[~valid] = ignore_index
    return out


class SegformerEdge(SegModel):
    """SegFormer with an edge branch, adapted from CE2P (Ruan et al., AAAI 2019) and SCHP.

    The unchanged SegFormer produces its usual fused decoder features. Three small
    branches read the first three encoder stages and are trained to predict garment
    and body-part outlines. Their features are concatenated with the decoder features,
    and a new classifier predicts the labels from both, so outline information reaches
    the final prediction.

    Training adds two terms to the main loss (CE2P: parsing + edge + edge-parsing):
      * class-balanced cross-entropy on the edge map (weights as in SCHP);
      * cross-entropy on SegFormer's own, pre-fusion prediction.
    At inference only the fused prediction is used.
    """

    EDGE_CHANNELS = 64

    def __init__(self, net: nn.Module, encoder: nn.Module, name: str, num_classes: int,
                 edge_weight: float = 1.0, aux_weight: float = 1.0) -> None:
        super().__init__(net, encoder, name)
        cfg = net.config
        c, d = self.EDGE_CHANNELS, cfg.decoder_hidden_size
        self.edge_weight, self.aux_weight = edge_weight, aux_weight
        self.edge_branches = nn.ModuleList(
            nn.Sequential(nn.Conv2d(h, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
            for h in cfg.hidden_sizes[:3]
        )
        self.edge_head = nn.Conv2d(3 * c, 2, kernel_size=3, padding=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(d + 3 * c, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            nn.Dropout(cfg.classifier_dropout_prob), nn.Conv2d(d, num_classes, 1),
        )
        self._decoder_features: torch.Tensor | None = None
        self._train_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        # The input of SegFormer's classifier is its fused decoder feature map.
        net.decode_head.classifier.register_forward_pre_hook(self._keep_decoder_features)

    def _keep_decoder_features(self, module: nn.Module, inputs: tuple) -> None:
        self._decoder_features = inputs[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x, output_hidden_states=True)
        stages, fused = out.hidden_states, self._decoder_features
        self._decoder_features = None
        if fused is None or stages is None or len(stages) < 3 or stages[0].ndim != 4:
            raise RuntimeError("This version of the transformers library does not expose SegFormer's stage "
                               "features in the form the edge model expects.")
        size = fused.shape[-2:]
        edge_feats = torch.cat(
            [F.interpolate(branch(f), size=size, mode="bilinear", align_corners=False)
             for branch, f in zip(self.edge_branches, stages[:3])], dim=1)
        logits = self.fusion(torch.cat([fused, edge_feats], dim=1))
        if self.training:
            self._train_outputs = (self.edge_head(edge_feats), out.logits)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def aux_loss(self, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]] | None:
        if self._train_outputs is None:
            return None
        edge_logits, pre_fusion_logits = self._train_outputs
        self._train_outputs = None
        size = target.shape[-2:]
        edge_logits = F.interpolate(edge_logits.float(), size=size, mode="bilinear", align_corners=False)
        pre_fusion_logits = F.interpolate(pre_fusion_logits.float(), size=size, mode="bilinear", align_corners=False)
        edges = edge_target(target)
        pos, neg = (edges == 1).sum().float(), (edges == 0).sum().float()
        total = (pos + neg).clamp(min=1)
        # The rarer class (edge) gets the larger weight. The floor keeps the loss finite
        # for a batch that happens to contain only one of the two classes.
        weights = torch.stack([pos / total, neg / total]).clamp(min=1e-4)
        edge = F.cross_entropy(edge_logits, edges, weight=weights, ignore_index=255)
        aux = F.cross_entropy(pre_fusion_logits, target, ignore_index=255)
        return self.edge_weight * edge + self.aux_weight * aux, {"edge": float(edge.detach()), "aux": float(aux.detach())}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> SegModel:
    """Build a candidate model by name (see ``MODEL_NAMES``).

    ``pretrained=False`` skips the ImageNet download; use it when the weights are
    about to be loaded from a checkpoint anyway.
    """
    edge = name.endswith("_edge")
    if name.startswith("segformer_"):
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        from transformers.utils import logging as hf_logging

        size = name.split("_")[1]
        # "_ade": start from SegFormer weights already trained for segmentation on ADE20K
        # (encoder and decoder); only the 150-class classifier is replaced. Otherwise the
        # encoder starts from ImageNet classification weights and the decoder is new.
        ade = name.endswith("_ade")
        repo = f"nvidia/segformer-{size}-finetuned-ade-512-512" if ade else f"nvidia/mit-{size}"
        if pretrained:
            # The checkpoint is an ImageNet classifier: its classifier is dropped and the
            # decode head is new. transformers prints a long report about that; silence it.
            previous = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                net = SegformerForSemanticSegmentation.from_pretrained(repo, num_labels=num_classes, ignore_mismatched_sizes=ade)
            finally:
                hf_logging.set_verbosity(previous)
        else:
            config = SegformerConfig.from_pretrained(repo)
            config.num_labels = num_classes
            net = SegformerForSemanticSegmentation(config)
        encoder = net.segformer
    elif name == "upernet_convnext_tiny":
        import segmentation_models_pytorch as smp

        net = smp.UPerNet(encoder_name="tu-convnext_tiny", encoder_weights="imagenet" if pretrained else None, classes=num_classes)
        encoder = net.encoder
    elif name == "deeplabv3plus_resnet101":
        import segmentation_models_pytorch as smp

        # The classic human-parsing baseline: ResNet-101 encoder, atrous spatial pyramid pooling decoder.
        net = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet" if pretrained else None, classes=num_classes)
        encoder = net.encoder
    elif name.startswith("dual_") and name.endswith("_convnext_tiny"):
        from .dual import build_dual

        # Two encoders (MiT + ConvNeXt-Tiny) with a fusion decoder; see dual.py.
        net = build_dual(name.split("_")[1], num_classes, pretrained)
        encoder = net.encoders()
    else:
        raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_NAMES)}")

    if edge:
        model = SegformerEdge(net, encoder, name, num_classes)
    elif name.startswith("dual_"):
        model = DualSegModel(net, encoder, name)
    else:
        model = SegModel(net, encoder, name)
    n_enc = sum(p.numel() for p in encoder.parameters()) / 1e6
    n_all = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {name}: {n_all:.1f} M parameters (encoder {n_enc:.1f} M, head {n_all - n_enc:.1f} M) | {num_classes} classes | "
          + (("encoder and decoder: ADE20K segmentation weights loaded, classifier: new" if name.endswith("_ade")
             else "both encoders: ImageNet-pretrained weights loaded, fusion and decoder: new" if name.startswith("dual_")
             else "encoder: ImageNet-pretrained weights loaded, head: new") if pretrained else "weights not initialised from ImageNet"), flush=True)
    return model


# Weight names of the Hugging Face SegFormer modules changed between transformers releases
# (checkpoints written with 5.18 use "stages.0.blocks.0.attention.q_proj", 5.5 uses
# "encoder.block.0.0.attention.self.query"). Same tensors, different names. Each pair maps a
# newer name (left) to the older one (right); they are applied as plain regular expressions.
_SEGFORMER_NEW_TO_OLD = (
    (r"\.stages\.(\d+)\.patch_embeddings\.", r".encoder.patch_embeddings.\1."),
    (r"\.stages\.(\d+)\.layer_norm\.", r".encoder.layer_norm.\1."),
    (r"\.stages\.(\d+)\.blocks\.(\d+)\.", r".encoder.block.\1.\2."),
    (r"\.layernorm_before\.", r".layer_norm_1."),
    (r"\.layernorm_after\.", r".layer_norm_2."),
    (r"\.attention\.q_proj\.", r".attention.self.query."),
    (r"\.attention\.k_proj\.", r".attention.self.key."),
    (r"\.attention\.v_proj\.", r".attention.self.value."),
    (r"\.attention\.o_proj\.", r".attention.output.dense."),
    (r"\.attention\.sequence_reduction\.sequence_reduction\.", r".attention.self.sr."),
    (r"\.attention\.sequence_reduction\.layer_norm\.", r".attention.self.layer_norm."),
    (r"\.mlp\.fc1\.", r".mlp.dense1."),
    (r"\.mlp\.fc2\.", r".mlp.dense2."),
    (r"\.decode_head\.linear_projections\.", r".decode_head.linear_c."),
)
_SEGFORMER_OLD_TO_NEW = (
    (r"\.encoder\.patch_embeddings\.(\d+)\.", r".stages.\1.patch_embeddings."),
    (r"\.encoder\.layer_norm\.(\d+)\.", r".stages.\1.layer_norm."),
    (r"\.encoder\.block\.(\d+)\.(\d+)\.", r".stages.\1.blocks.\2."),
    (r"\.layer_norm_1\.", r".layernorm_before."),
    (r"\.layer_norm_2\.", r".layernorm_after."),
    (r"\.attention\.self\.query\.", r".attention.q_proj."),
    (r"\.attention\.self\.key\.", r".attention.k_proj."),
    (r"\.attention\.self\.value\.", r".attention.v_proj."),
    (r"\.attention\.output\.dense\.", r".attention.o_proj."),
    (r"\.attention\.self\.sr\.", r".attention.sequence_reduction.sequence_reduction."),
    (r"\.attention\.self\.layer_norm\.", r".attention.sequence_reduction.layer_norm."),
    (r"\.mlp\.dense1\.", r".mlp.fc1."),
    (r"\.mlp\.dense2\.", r".mlp.fc2."),
    (r"\.decode_head\.linear_c\.", r".decode_head.linear_projections."),
)


def _rename(key: str, rules: tuple[tuple[str, str], ...]) -> str:
    import re

    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def load_weights(model: nn.Module, state: dict) -> None:
    """``model.load_state_dict(state)``, also across transformers versions that renamed SegFormer's weights.

    The names are used as they are when they match. Otherwise each name that the model does
    not know is translated to the other naming scheme, and the result is used only if every
    name and every tensor shape then matches exactly; if not, the usual error is raised.
    """
    target = model.state_dict()
    if set(state) == set(target):
        model.load_state_dict(state)
        return
    for rules in (_SEGFORMER_NEW_TO_OLD, _SEGFORMER_OLD_TO_NEW):
        renamed = {(k if k in target else _rename(k, rules)): v for k, v in state.items()}
        if set(renamed) == set(target) and all(renamed[k].shape == target[k].shape for k in target):
            model.load_state_dict(renamed)
            print(f"[model] weight names translated between transformers versions ({sum(k not in target for k in state)} names)", flush=True)
            return
    model.load_state_dict(state)  # raises the standard, detailed error

