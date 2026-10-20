"""Train a clothes-segmentation model, or resume it. The same command does both.

If the run directory already holds a checkpoint, training continues from it (also
from the middle of an epoch). Otherwise it starts fresh. The command stops cleanly
and saves when the time budget is reached; run it again to go on.

Examples (Colab):
    # Bake-off round 1: 12-epoch schedule, stop after 5 epochs
    python scripts/train.py --drive-root /content/drive/MyDrive/clothseg \
        --run-name r1_segformer_b2 --model segformer_b2 --epochs 12 --stop-after-epoch 5

    # Round 2: the same command with a later stop resumes at epoch 6
    python scripts/train.py --drive-root /content/drive/MyDrive/clothseg \
        --run-name r1_segformer_b2 --model segformer_b2 --epochs 12
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clothseg.config import TrainConfig  # noqa: E402
from clothseg.data.preprocess import DONE_MARKER, cache_name  # noqa: E402
from clothseg.engine import Runtime, train  # noqa: E402
from clothseg.models import MODEL_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("where")
    g.add_argument("--drive-root", type=Path, required=True, help="Persistent project folder. Runs are stored in <drive-root>/runs/<run-name>.")
    g.add_argument("--run-name", required=True, help="Name of the experiment. Reusing a name resumes that run.")
    g.add_argument("--local-root", type=Path, default=Path("/content/data"), help="Folder that holds the extracted dataset cache.")
    g.add_argument("--extend-from", default="", help="Name of a FINISHED run to continue with a longer schedule (--epochs larger than its own). "
                   "Used only when --run-name has no checkpoint yet; that run itself is only read.")
    g = p.add_argument_group("what is trained (fixed for the life of a run)")
    g.add_argument("--model", default=d.model, choices=MODEL_NAMES)
    g.add_argument("--num-classes", type=int, default=d.num_classes, choices=(15, 18))
    g.add_argument("--height", type=int, default=d.height)
    g.add_argument("--width", type=int, default=d.width)
    g.add_argument("--epochs", type=int, default=d.epochs, help="Length of the learning-rate schedule in epochs.")
    g.add_argument("--batch-size", type=int, default=d.batch_size)
    g.add_argument("--encoder-lr", type=float, default=d.encoder_lr)
    g.add_argument("--head-lr", type=float, default=d.head_lr)
    g.add_argument("--weight-decay", type=float, default=d.weight_decay)
    g.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    g.add_argument("--ce-weight", type=float, default=d.ce_weight)
    g.add_argument("--lovasz-weight", type=float, default=d.lovasz_weight)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--no-amp", action="store_true", help="Disable fp16 mixed precision.")
    g.add_argument("--no-pretrained", action="store_true", help="Random encoder weights (tests only).")
    g.add_argument("--no-augment", action="store_true", help="Train without data augmentation (ablation).")
    g.add_argument("--limit-train", type=int, default=0, help="Use only the first N training images (tests only).")
    g.add_argument("--limit-val", type=int, default=0)
    g = p.add_argument_group("this session (may change between sessions)")
    g.add_argument("--time-budget-min", type=float, default=280.0, help="Stop cleanly and save after this many minutes (280 = 4h40m).")
    g.add_argument("--save-every-min", type=float, default=10.0, help="Minutes between checkpoints inside an epoch.")
    g.add_argument("--stop-after-epoch", type=int, default=0, help="Pause after this epoch even though the schedule is longer.")
    g.add_argument("--stop-after-steps", type=int, default=0, help="Pause after this global step (used by the resume test).")
    g.add_argument("--workers", type=int, default=-1, help="DataLoader workers. -1 = automatic from the CPU count.")
    g.add_argument("--log-every", type=int, default=50, help="Steps between log lines.")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cache_dir = a.local_root / cache_name(a.height, a.width)
    if not (cache_dir / DONE_MARKER).exists():
        sys.exit(f"ERROR: no dataset cache at {cache_dir}. Run scripts/prepare_data.py first (the 'Prepare the dataset' cell).")
    if not a.drive_root.exists():
        sys.exit(f"ERROR: {a.drive_root} does not exist. Mount Google Drive first.")
    cores = os.cpu_count() or 2
    workers = a.workers if a.workers >= 0 else (2 if cores <= 2 else min(8, cores - 2))

    cfg = TrainConfig(
        model=a.model, num_classes=a.num_classes, height=a.height, width=a.width, epochs=a.epochs, batch_size=a.batch_size,
        encoder_lr=a.encoder_lr, head_lr=a.head_lr, weight_decay=a.weight_decay, warmup_steps=a.warmup_steps,
        ce_weight=a.ce_weight, lovasz_weight=a.lovasz_weight, seed=a.seed, amp=not a.no_amp, pretrained=not a.no_pretrained, augment=not a.no_augment,
        limit_train=a.limit_train, limit_val=a.limit_val,
    )
    rt = Runtime(
        run_dir=a.drive_root / "runs" / a.run_name, cache_dir=cache_dir, workers=workers, time_budget_min=a.time_budget_min,
        save_every_min=a.save_every_min, log_every=a.log_every, stop_after_epoch=a.stop_after_epoch, stop_after_steps=a.stop_after_steps,
        extend_from=a.drive_root / "runs" / a.extend_from if a.extend_from else None,
    )
    try:
        train(cfg, rt)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            sys.exit(f"ERROR: the GPU ran out of memory (batch size {a.batch_size}). The last checkpoint is intact. "
                     "Use a GPU with more memory and run the same command again to continue. Only if that is not possible: "
                     f"start a NEW run name with --batch-size {max(1, a.batch_size // 2)}; such a run is not comparable with batch-{a.batch_size} runs.")
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
