"""PyTorch dataset and sampler for the preprocessed ATR cache.

Resume design: the order of an epoch and the augmentation of every sample are pure
functions of (seed, epoch, position). Nothing depends on hidden random state, so a
run that restarts in the middle of an epoch continues with exactly the batches it
would have seen without the interruption.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from ..labels import build_lut
from .splits import load_splits
from .transforms import letterbox, normalize, train_transform


class AtrDataset(Dataset):
    """Samples of one split of the cache.

    Training items are requested with a key ``epoch * len(dataset) + position`` (see
    :class:`ResumableBatchSampler`), which carries the epoch into the worker processes.
    Evaluation items are requested with a plain position.
    """

    def __init__(self, cache_dir: Path, split: str, num_classes: int, height: int, width: int,
                 train: bool, seed: int = 0, limit: int | None = None, augment: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.ids: list[int] = load_splits(self.cache_dir)[split]
        if limit:
            self.ids = self.ids[:limit]
        if not self.ids:
            raise RuntimeError(f"Split '{split}' of {self.cache_dir} is empty.")
        self.num_classes, self.height, self.width = num_classes, height, width
        self.train, self.seed, self.augment = train, seed, augment
        self.lut = build_lut(num_classes)

    def __len__(self) -> int:
        return len(self.ids)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        name = f"{idx:06d}"
        image = np.asarray(Image.open(self.cache_dir / "images" / f"{name}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(self.cache_dir / "masks" / f"{name}.png"))
        return image, mask

    def __getitem__(self, key: int) -> dict:
        cv2.setNumThreads(0)  # one thread per worker process; the workers already use all cores
        if self.train:
            epoch, pos = divmod(key, len(self.ids))
            idx = self.ids[pos]
            image, mask = self._load(idx)
            if self.augment:
                rng = np.random.default_rng([self.seed, epoch, idx])
                image, mask = train_transform(image, mask, self.height, self.width, rng, swap_left_right=self.num_classes == 18)
            else:  # ablation: the photo is only resized and padded, exactly as in evaluation
                image, mask, _ = letterbox(image, mask, self.height, self.width)
        else:
            idx = self.ids[key]
            image, mask = self._load(idx)
            image, mask, _ = letterbox(image, mask, self.height, self.width)
        return {
            "image": torch.from_numpy(normalize(image)),
            "mask": torch.from_numpy(self.lut[mask].astype(np.int64)),
            "idx": idx,
        }


class ResumableBatchSampler(Sampler[list[int]]):
    """Shuffled batches whose order depends only on (seed, epoch).

    Call :meth:`set_epoch` before iterating. ``start_batch`` skips the batches that
    were already trained on before an interruption.
    """

    def __init__(self, dataset_len: int, batch_size: int, seed: int) -> None:
        self.n, self.batch_size, self.seed = dataset_len, batch_size, seed
        self.epoch, self.start_batch = 0, 0

    @property
    def batches_per_epoch(self) -> int:
        return self.n // self.batch_size  # the incomplete last batch is dropped

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch

    def __iter__(self):
        perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        offset = self.epoch * self.n
        for b in range(self.start_batch, self.batches_per_epoch):
            yield [int(offset + p) for p in perm[b * self.batch_size:(b + 1) * self.batch_size]]
