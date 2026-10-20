"""Train/val/test split handling for the preprocessed cache.

``splits.json`` is written once by the preprocessing step. This module loads it and
removes leakage: near-duplicate photos (same perceptual hash) that sit in more than
one split are all moved into train, so validation and test never contain a copy of
a training image.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
ORIGINAL_NAME = "splits_original.json"


def load_splits(cache_dir: Path) -> dict:
    """Read ``splits.json`` of a cache directory."""
    with open(Path(cache_dir) / "splits.json") as f:
        return json.load(f)


def load_meta(cache_dir: Path) -> dict[int, dict]:
    """Read ``meta.csv`` as ``{sample idx: row}``."""
    with open(Path(cache_dir) / "meta.csv", newline="") as f:
        return {int(r["idx"]): r for r in csv.DictReader(f)}


def move_cross_split_near_duplicates(cache_dir: Path) -> dict:
    """Move near-duplicates that span several splits into train. Safe to call repeatedly.

    The untouched file is kept as ``splits_original.json``. The result is fully
    determined by ``meta.csv`` and the original splits, so every session that
    extracts the cache again ends up with exactly the same splits.
    """
    cache_dir = Path(cache_dir)
    splits = load_splits(cache_dir)
    if splits.get("near_duplicates_moved_to_train") is not None:
        sizes = ", ".join(f"{s}={len(splits[s])}" for s in SPLITS)
        print(f"[splits] cross-split near-duplicates already moved into train. Sizes: {sizes}", flush=True)
        return splits

    meta = load_meta(cache_dir)
    where = {i: s for s in SPLITS for i in splits[s]}
    groups: dict[str, list[int]] = defaultdict(list)
    for i in where:
        groups[meta[i]["dhash"]].append(i)

    moved = sorted(
        i for members in groups.values()
        if len({where[m] for m in members}) > 1
        for i in members if where[i] != "train"
    )
    moved_set = set(moved)

    (cache_dir / ORIGINAL_NAME).write_text(json.dumps(splits))
    for s in ("val", "test"):
        splits[s] = [i for i in splits[s] if i not in moved_set]
    splits["train"] = sorted(splits["train"] + moved)
    splits["near_duplicates_moved_to_train"] = moved
    tmp = cache_dir / "splits.json.tmp"
    tmp.write_text(json.dumps(splits))
    tmp.replace(cache_dir / "splits.json")
    sizes = ", ".join(f"{s}={len(splits[s])}" for s in SPLITS)
    print(f"[splits] moved {len(moved)} val/test near-duplicates of training images into train. New sizes: {sizes}", flush=True)
    return splits
