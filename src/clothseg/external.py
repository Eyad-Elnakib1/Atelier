"""Other segmentation models behind the same interface as ``Segmenter``, for fair comparisons.

Each wrapper takes an RGB uint8 photo of any size and returns a label map with the photo's
height and width. ``YoloSemSegmenter`` and ``SchpAtr`` return this project's 15 classes;
``FashnParser`` returns its own 18 classes (mapped by the comparison code, which knows
which classes can be compared). ``score_atr`` scores any of them on an ATR split exactly
like ``scripts/evaluate.py``: the cached photo is brought to the original size, predicted,
and compared with the full-size mask.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .labels import ATR_TO_15, build_lut
from .metrics import ConfusionMatrix

# ATR's 18 ids -> this project's 15 classes (left/right merged); anything else -> ignore.
_ATR18_TO_15 = np.array(list(ATR_TO_15) + [255] * (256 - len(ATR_TO_15)), dtype=np.uint8)


class YoloSemSegmenter:
    """An Ultralytics YOLO semantic-segmentation model trained on this project's 15 classes."""

    def __init__(self, weights: str | Path, imgsz: int = 576, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        # Ultralytics reads numpy images as BGR, like OpenCV.
        r = self.model.predict(np.ascontiguousarray(image[:, :, ::-1]), imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.semantic_mask is None:
            raise RuntimeError("The YOLO model returned no semantic mask; is it a '-sem' model?")
        labels = r.semantic_mask.data.cpu().numpy().astype(np.uint8)
        if labels.shape != image.shape[:2]:
            labels = cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return labels


class SchpAtr:
    """SCHP (Self-Correction for Human Parsing) trained on ATR, from the Hugging Face Hub; returns 15 classes."""

    REPO = "pirocheto/schp-atr-18"
    # The model runs its own code from the Hub (trust_remote_code); pinned to the revision that was checked.
    REVISION = "61f4ff3c508b899904e495cefbbaff9cacadeb45"

    def __init__(self, device: str | None = None) -> None:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(self.REPO, revision=self.REVISION, trust_remote_code=True)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(self.REPO, revision=self.REVISION, trust_remote_code=True).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits  # the head used in the model card's own example
        logits = F.interpolate(logits.float(), size=image.shape[:2], mode="bilinear", align_corners=False)
        return _ATR18_TO_15[logits.argmax(1)[0].to(torch.uint8).cpu().numpy()]


class FashnParser:
    """FASHN Human Parser (SegFormer-B4) from the Hugging Face Hub; returns FASHN's own 18 class ids."""

    REPO = "fashn-ai/fashn-human-parser"
    REVISION = "1f80c34dbab321c5730dda5c3fea279fd3e97498"

    def __init__(self, device: str | None = None) -> None:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # use_fast=False: the PIL processor keeps the LANCZOS resizing of the model card.
        self.processor = AutoImageProcessor.from_pretrained(self.REPO, revision=self.REVISION, use_fast=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.REPO, revision=self.REVISION).to(self.device).eval()
        self.names = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        logits = F.interpolate(logits.float(), size=image.shape[:2], mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].to(torch.uint8).cpu().numpy()


def score_atr(predict, cache: str | Path, ids: list[int], device: str | None = None, diagnosis=None) -> dict:
    """Score a 15-class predictor on ATR photos at the original resolution, like scripts/evaluate.py.

    ``diagnosis`` (a clothseg.diagnose.Diagnosis over the 15 classes), if given, also collects where the errors are.
    """
    cache = Path(cache)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lut = build_lut(15)
    cm = ConfusionMatrix(15, dev)
    for idx in ids:
        image = np.asarray(Image.open(cache / "images" / f"{idx:06d}.jpg").convert("RGB"))
        gt = lut[np.asarray(Image.open(cache / "masks_orig" / f"{idx:06d}.png"))]
        full = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) if image.shape[:2] != gt.shape else image
        pred = predict(full)
        cm.update(torch.from_numpy(pred).to(dev), torch.from_numpy(gt).to(dev))
        if diagnosis is not None:
            diagnosis.update(gt, pred, idx)
    m = cm.compute()
    m["core_garment_miou"] = float(np.mean([m["iou"][n] for n in ("upper-clothes", "pants", "skirt", "dress")]))
    return m
