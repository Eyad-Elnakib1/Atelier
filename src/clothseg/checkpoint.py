"""Crash-proof checkpoints on Google Drive.

Rules that make a disconnect harmless:
  * A checkpoint is first written to the local disk, then copied to Drive under a
    temporary name and renamed. A file with the final name is therefore always complete.
  * The two most recent checkpoints are kept. If the newest cannot be read, the
    previous one is used.
  * ``best.pt`` is separate and only replaced when the validation metric improves.
"""
from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data.storage import atomic_copy

_PATTERN = re.compile(r"^step_(\d{8})\.pt$")
KEEP_LAST = 2


def rng_state() -> dict:
    """Snapshot of every random generator the training loop touches."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as exc:  # noqa: BLE001 - e.g. resumed on a machine with another GPU count
            print(f"[checkpoint] WARNING: could not restore the CUDA random state ({exc}). Training continues.", flush=True)


def _save(state: dict, dst: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / dst.name
        torch.save(state, local)
        atomic_copy(local, dst, quiet=True)


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Step checkpoints of a run, oldest first."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted(p for p in ckpt_dir.iterdir() if _PATTERN.match(p.name))


def save_checkpoint(run_dir: Path, state: dict) -> Path:
    """Save ``state`` as ``checkpoints/step_<global_step>.pt`` and delete older ones beyond ``KEEP_LAST``."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{state['global_step']:08d}.pt"
    _save(state, path)
    for old in list_checkpoints(run_dir)[:-KEEP_LAST]:
        old.unlink(missing_ok=True)
    for stale in ckpt_dir.glob("*.tmp"):  # left behind by a session that died while copying
        stale.unlink(missing_ok=True)
    return path


def save_best(run_dir: Path, state: dict) -> Path:
    """Save the best model so far as ``checkpoints/best.pt`` (model weights, metrics and config only)."""
    path = Path(run_dir) / "checkpoints" / "best.pt"
    _save({k: state[k] for k in ("model", "config", "epoch", "global_step", "best_metric", "val_metrics")}, path)
    return path


def load_latest(run_dir: Path) -> dict | None:
    """Load the newest readable checkpoint of a run, or return ``None`` if there is none."""
    for path in reversed(list_checkpoints(run_dir)):
        try:
            # weights_only=False: the file holds optimizer state, numpy RNG state and the
            # config. It was written by this code, so unpickling it is safe.
            state = torch.load(path, map_location="cpu", weights_only=False)
            missing = {"model", "optimizer", "scaler", "epoch", "batch_in_epoch", "global_step", "rng", "config"} - state.keys()
            if missing:
                raise KeyError(f"missing entries {sorted(missing)}")
            print(f"[checkpoint] loaded {path.name}", flush=True)
            return state
        except Exception as exc:  # noqa: BLE001 - a damaged file must not stop the resume
            print(f"[checkpoint] WARNING: {path.name} is unreadable ({type(exc).__name__}: {exc}). Trying the previous one.", flush=True)
    return None
