"""Image/mask transforms for training and evaluation.

All randomness comes from a ``numpy.random.Generator`` handed in by the dataset. The
dataset seeds it from (run seed, epoch, sample index), so an interrupted run that
resumes mid-epoch applies exactly the augmentations it would have applied anyway.

Geometry (flip, scale, rotation, shift) is one affine warp, so the image is only
interpolated once. Pixels that fall outside the photo get the label ``IGNORE_INDEX``
and do not contribute to the loss.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..labels import FLIP_SWAP_18, IGNORE_INDEX

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_FLIP_LUT_18 = np.arange(256, dtype=np.uint8)
for _a, _b in FLIP_SWAP_18:
    _FLIP_LUT_18[_a], _FLIP_LUT_18[_b] = _b, _a


def normalize(image: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB -> float32 3xHxW, ImageNet-normalised."""
    x = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def letterbox(image: np.ndarray, mask: np.ndarray | None, height: int, width: int) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]:
    """Resize to fit inside (height, width) keeping the aspect ratio, then centre-pad.

    Returns the padded image, the padded mask (padding = ``IGNORE_INDEX``) and the
    box ``(top, left, h, w)`` of the photo inside the padded canvas.
    """
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) != (h, w):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    top, left = (height - nh) // 2, (width - nw) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = image
    out_mask = None
    if mask is not None:
        out_mask = np.full((height, width), IGNORE_INDEX, dtype=np.uint8)
        out_mask[top:top + nh, left:left + nw] = mask
    return canvas, out_mask, (top, left, nh, nw)


def _photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lighting, colour and camera-quality changes. Classes are defined by shape, so colour may move."""
    if rng.random() < 0.8:  # brightness / contrast
        alpha, beta = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # gamma: under/over-exposure
        gamma = rng.uniform(0.7, 1.5)
        table = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        image = cv2.LUT(image, table)
    if rng.random() < 0.5:  # mild hue / saturation shift
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-8, 9))) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.15:  # out-of-focus / motion softness
        k = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (k, k), 0)
    if rng.random() < 0.15:  # sensor noise
        noise = rng.normal(0, rng.uniform(3, 10), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:  # JPEG artefacts of messaging apps
        ok, enc = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
        if ok:
            image = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return image


def train_transform(image: np.ndarray, mask: np.ndarray, height: int, width: int,
                    rng: np.random.Generator, swap_left_right: bool) -> tuple[np.ndarray, np.ndarray]:
    """Random flip, scale 0.5-1.5, rotation up to 15 degrees, random placement, then photometric changes.

    ``swap_left_right`` must be True for the 18-class label space: ATR's left/right
    labels follow the person, so a mirrored image needs them exchanged.
    """
    h, w = image.shape[:2]
    fit = min(height / h, width / w)                 # scale at which the photo exactly fits the canvas
    scale = fit * rng.uniform(0.5, 1.5)
    angle = rng.uniform(-15, 15) if rng.random() < 0.5 else 0.0
    flip = rng.random() < 0.5

    # Free room (scale < fit) or overflow (scale > fit): either way pick a random placement.
    slack_x, slack_y = abs(width - w * scale) / 2, abs(height - h * scale) / 2
    shift_x, shift_y = rng.uniform(-slack_x, slack_x), rng.uniform(-slack_y, slack_y)

    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)   # rotate/scale about the photo centre
    m[0, 2] += width / 2 - w / 2 + shift_x
    m[1, 2] += height / 2 - h / 2 + shift_y
    if flip:
        image, mask = image[:, ::-1], mask[:, ::-1]
        if swap_left_right:
            mask = _FLIP_LUT_18[mask]
        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)

    image = cv2.warpAffine(image, m, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, m, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE_INDEX)
    image = _photometric(image, rng)
    return image, mask
