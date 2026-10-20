"""Label definitions for the ATR human-parsing dataset and our training label spaces.

The preprocessed cache always stores the original 18 ATR ids. The mapping to the
training label space (15 or 18 classes) is applied when a sample is loaded, so one
cache serves both experiments.
"""
from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255

# Original ATR label ids, as listed on the dataset card.
ATR_CLASSES: tuple[str, ...] = (
    "background", "hat", "hair", "sunglasses", "upper-clothes", "skirt", "pants", "dress", "belt",
    "left-shoe", "right-shoe", "face", "left-leg", "right-leg", "left-arm", "right-arm", "bag", "scarf",
)

# 15-class space: left/right shoes, legs and arms are merged.
CLASSES_15: tuple[str, ...] = (
    "background", "hat", "hair", "sunglasses", "upper-clothes", "skirt", "pants", "dress", "belt",
    "shoes", "face", "legs", "arms", "bag", "scarf",
)

# ATR id -> 15-class id (index = ATR id).
ATR_TO_15: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 11, 12, 12, 13, 14)

# ATR id pairs that must be swapped when an 18-class sample is mirrored.
FLIP_SWAP_18: tuple[tuple[int, int], ...] = ((9, 10), (12, 13), (14, 15))

# Clothing and accessory classes, used for the garment-only mIoU.
GARMENT_NAMES: frozenset[str] = frozenset(
    {"hat", "sunglasses", "upper-clothes", "skirt", "pants", "dress", "belt",
     "shoes", "left-shoe", "right-shoe", "bag", "scarf"}
)

_PALETTE_18 = np.array(
    [[0, 0, 0], [128, 0, 0], [255, 0, 0], [170, 0, 51], [255, 85, 0], [0, 128, 0], [0, 85, 85], [0, 0, 85],
     [85, 51, 0], [255, 255, 0], [255, 170, 0], [0, 0, 255], [85, 255, 170], [170, 255, 85], [51, 170, 221],
     [0, 255, 255], [0, 119, 221], [52, 86, 128]],
    dtype=np.uint8,
)


def class_names(num_classes: int) -> tuple[str, ...]:
    """Return the class names of the 15- or 18-class label space."""
    if num_classes == 18:
        return ATR_CLASSES
    if num_classes == 15:
        return CLASSES_15
    raise ValueError(f"num_classes must be 15 or 18, got {num_classes}")


def garment_ids(num_classes: int) -> list[int]:
    """Return the ids of clothing/accessory classes in the given label space."""
    return [i for i, n in enumerate(class_names(num_classes)) if n in GARMENT_NAMES]


def build_lut(num_classes: int) -> np.ndarray:
    """Build a 256-entry lookup table from stored ATR ids to training ids.

    Any value that is not a valid ATR id maps to ``IGNORE_INDEX``.
    """
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    if num_classes == 18:
        lut[:18] = np.arange(18, dtype=np.uint8)
    elif num_classes == 15:
        lut[:18] = np.array(ATR_TO_15, dtype=np.uint8)
    else:
        raise ValueError(f"num_classes must be 15 or 18, got {num_classes}")
    return lut


def palette(num_classes: int) -> np.ndarray:
    """Return an (num_classes, 3) uint8 RGB palette."""
    if num_classes == 18:
        return _PALETTE_18.copy()
    if num_classes == 15:
        # Reuse the colour of the first ATR class that maps to each merged class.
        first = [ATR_TO_15.index(i) for i in range(15)]
        return _PALETTE_18[first].copy()
    raise ValueError(f"num_classes must be 15 or 18, got {num_classes}")
