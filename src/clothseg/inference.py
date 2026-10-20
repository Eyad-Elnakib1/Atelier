"""Inference: turn any photo of a person into a label mask, an overlay and garment cut-outs.

Python API::

    from clothseg.inference import Segmenter, load_image, save_outputs

    seg = Segmenter("best.pt")
    image = load_image("photo.jpg")          # EXIF rotation applied, converted to RGB
    mask = seg.predict(image)                # (H, W) uint8 class ids at the photo's own size
    save_outputs(image, mask, seg, "out", "photo")

The same ``Segmenter.predict`` is used by ``scripts/evaluate.py``, so the reported
metrics describe exactly the pipeline that is shipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .config import TrainConfig
from .data.transforms import letterbox, normalize
from .labels import FLIP_SWAP_18, class_names, garment_ids, palette
from .models import build_model, load_weights

# Logits are upsampled to at most this many pixels on the long side. Larger photos get
# their label map enlarged with nearest neighbour, which keeps memory bounded.
MAX_LOGIT_SIDE = 1536


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array, honouring the EXIF orientation tag.

    Transparent images are placed on white; grayscale, palette and CMYK images are
    converted to RGB.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            background = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, im)
        return np.asarray(im.convert("RGB"))


class Segmenter:
    """A trained model plus the pre- and post-processing that goes with it."""

    def __init__(self, checkpoint: str | Path, device: str | None = None, amp: bool = True, flip_tta: bool = False,
                 scales: tuple[float, ...] = (1.0,)) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        # weights_only=False: the file also holds the training config (a plain dict). Only load
        # checkpoints you created or trust.
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - any unreadable file gets the same clear message
            raise ValueError(f"{checkpoint} could not be read as a PyTorch checkpoint ({type(exc).__name__}).") from exc
        if not isinstance(state, dict) or "model" not in state or "config" not in state:
            raise ValueError(f"{checkpoint} is not a checkpoint written by this project .")
        self.cfg = TrainConfig.from_dict(state["config"])
        self.epoch = state.get("epoch")
        self.num_classes = self.cfg.num_classes
        self.names = class_names(self.num_classes)
        self.palette = palette(self.num_classes)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = amp and self.device.type == "cuda"
        # Flip test-time augmentation: also predict on the mirrored photo and average the two.
        self.flip_tta = flip_tta
        # Multi-scale test-time augmentation: predict at several input sizes and average.
        self.scales = tuple(scales)
        if not self.scales or min(self.scales) <= 0:
            raise ValueError(f"scales must be positive numbers, got {scales}")
        self.model = build_model(self.cfg.model, self.num_classes, pretrained=False)
        load_weights(self.model, state["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Segment one RGB uint8 image of any size. Returns class ids with the image's height and width."""
        return self._labels(self.scores(image), image.shape[:2])

    @staticmethod
    def _labels(scores: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
        """Score map (1, C, h, w) -> uint8 label mask at the photo's size."""
        h, w = size
        labels = scores.argmax(1)[0].to(torch.uint8).cpu().numpy()
        if labels.shape != (h, w):
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        return labels

    @torch.no_grad()
    def scores(self, image: np.ndarray) -> torch.Tensor:
        """Class scores (1, C, h, w) for one RGB uint8 image, averaged over flip and scales if enabled.

        The map has the photo's size, or a reduced size for very large photos.
        """
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected an RGB uint8 array of shape (H, W, 3); got {image.shape} {image.dtype}. Use load_image().")
        h, w = image.shape[:2]
        if min(h, w) < 2:
            raise ValueError(f"Image is too small to segment: {w}x{h} pixels.")
        scale = min(1.0, MAX_LOGIT_SIDE / max(h, w))
        th, tw = max(1, round(h * scale)), max(1, round(w * scale))
        total = None
        for s in self.scales:
            # The network needs sides divisible by 32.
            net_h = max(32, round(self.cfg.height * s / 32) * 32)
            net_w = max(32, round(self.cfg.width * s / 32) * 32)
            boxed, _, (top, left, nh, nw) = letterbox(image, None, net_h, net_w)
            x = torch.from_numpy(normalize(boxed)).unsqueeze(0).to(self.device)
            logits = self._logits(x)
            if self.flip_tta:
                mirrored = self._logits(x.flip(-1)).flip(-1)
                if self.num_classes == 18:
                    # In the mirrored photo the person's left side is on the other side, so the
                    # left/right channels have to be exchanged before averaging.
                    order = list(range(18))
                    for a, b in FLIP_SWAP_18:
                        order[a], order[b] = b, a
                    mirrored = mirrored[:, order]
                logits = (logits + mirrored) / 2
            logits = logits[:, :, top:top + nh, left:left + nw]  # drop the padding
            logits = F.interpolate(logits, size=(th, tw), mode="bilinear", align_corners=False)
            total = logits if total is None else total + logits
        return total / len(self.scales)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Float32 class scores of one normalised batch."""
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.model(x)
        if self.amp and not torch.isfinite(logits).all():
            logits = self.model(x)  # rare fp16 overflow: redo in float32
        return logits.float()

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        """Label mask -> RGB colour image."""
        return self.palette[mask]

    def overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Photo with the class colours blended on top; the background is dimmed instead of coloured."""
        out = (1 - alpha) * image.astype(np.float32) + alpha * self.colorize(mask).astype(np.float32)
        out[mask == 0] = image[mask == 0] * 0.6
        return out.astype(np.uint8)

    def cutouts(self, image: np.ndarray, mask: np.ndarray, min_area: float = 0.002) -> dict[str, Image.Image]:
        """One transparent RGBA image per garment found, cropped to the garment's bounding box.

        Garments covering less than ``min_area`` of the photo are skipped as likely noise.
        """
        result = {}
        for cid in garment_ids(self.num_classes):
            region = mask == cid
            if region.mean() < min_area:
                continue
            ys, xs = np.where(region)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            rgba = np.dstack([image[y0:y1, x0:x1], region[y0:y1, x0:x1].astype(np.uint8) * 255])
            result[self.names[cid]] = Image.fromarray(rgba)  # 4 uint8 channels: RGBA
        return result


class EnsembleSegmenter(Segmenter):
    """Averages the class probabilities of several trained models.

    The members may have different architectures, but must use the same classes. It
    offers the same methods as :class:`Segmenter`, so the scripts can use either.
    Prediction time and memory are the sum of the members.
    """

    def __init__(self, members: list[Segmenter]) -> None:  # noqa: D107 - does not load a checkpoint itself
        if not members:
            raise ValueError("An ensemble needs at least one model.")
        if len({m.num_classes for m in members}) != 1:
            raise ValueError(f"All models of an ensemble must have the same classes; got {[m.num_classes for m in members]}.")
        first = members[0]
        self.members = members
        self.cfg, self.epoch = first.cfg, None
        self.num_classes, self.names, self.palette, self.device = first.num_classes, first.names, first.palette, first.device

    @torch.no_grad()
    def scores(self, image: np.ndarray) -> torch.Tensor:
        # Probabilities, not raw scores: different architectures produce scores on different scales.
        total = None
        for member in self.members:
            p = member.scores(image).softmax(dim=1)
            total = p if total is None else total + p
        return total / len(self.members)


def save_outputs(image: np.ndarray, mask: np.ndarray, seg: Segmenter, out_dir: str | Path, stem: str,
                 min_area: float = 0.002) -> dict:
    """Write ``<stem>_mask.png``, ``<stem>_overlay.jpg``, ``<stem>_<garment>.png`` and ``<stem>.json``.

    The mask is a palette PNG: pixel values are the class ids, and image viewers show it in colour.
    Returns the summary that is also written to the JSON file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # frombytes, not fromarray(mask, "P"): the mode argument of fromarray is removed in Pillow 13.
    mask_img = Image.frombytes("P", (mask.shape[1], mask.shape[0]), np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    mask_img.putpalette(seg.palette.flatten().tolist())
    mask_img.save(out_dir / f"{stem}_mask.png")
    Image.fromarray(seg.overlay(image, mask)).save(out_dir / f"{stem}_overlay.jpg", quality=92)
    cut = seg.cutouts(image, mask, min_area)
    for name, rgba in cut.items():
        rgba.save(out_dir / f"{stem}_{name}.png")
    counts = np.bincount(mask.ravel(), minlength=seg.num_classes)
    summary = {
        "width": int(image.shape[1]), "height": int(image.shape[0]),
        "model": seg.cfg.model, "classes": {i: n for i, n in enumerate(seg.names)},
        "area_percent": {seg.names[i]: round(100 * int(c) / mask.size, 3) for i, c in enumerate(counts) if c},
        "cutouts": sorted(cut),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    return summary
