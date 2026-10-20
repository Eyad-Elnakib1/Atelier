"""Training configuration.

``TrainConfig`` holds everything that defines *what* is trained. It is saved as
``config.json`` in the run directory and inside every checkpoint. Things that only
affect *how* a session runs (time budget, worker count, paths) are not part of it,
so they can change between sessions without invalidating a run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    model: str = "segformer_b2"
    num_classes: int = 15            # 15 (left/right merged) or 18 (original ATR)
    height: int = 576
    width: int = 384
    epochs: int = 12                 # length of the learning-rate schedule
    batch_size: int = 16
    encoder_lr: float = 6e-5         # SegFormer paper
    head_lr: float = 6e-4            # 10x: the head starts from random values
    weight_decay: float = 0.01
    warmup_steps: int = 1500
    ce_weight: float = 0.5
    lovasz_weight: float = 0.5
    seed: int = 42
    amp: bool = True                 # fp16 mixed precision
    pretrained: bool = True          # ImageNet encoder weights
    augment: bool = True             # False: training photos are only resized (ablation)
    limit_train: int = 0             # >0: use only the first N training images (tests)
    limit_val: int = 0
    data_fingerprint: str = ""       # filled in at start: guards against a changed dataset

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def differences(self, other: "TrainConfig") -> list[str]:
        """Human-readable list of fields whose values differ."""
        a, b = self.to_dict(), other.to_dict()
        return [f"{k}: run has {a[k]!r}, command asks for {b[k]!r}" for k in a if a[k] != b[k]]
