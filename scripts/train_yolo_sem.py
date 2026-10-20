"""Train an Ultralytics YOLO26 semantic-segmentation model on this project's ATR split.

Used to compare YOLO26-sem with SegFormer-B3 on the same photos: the same training and
validation split (after the near-duplicate fix), the same 15 classes. YOLO is trained with
its own recipe (loss, optimizer, augmentation); only the number of epochs, the image size,
the batch size and the seed are set here, to match the SegFormer comparison runs.

The dataset is built from the project's cache (scripts/prepare_data.py) as hard links, so
no photo is copied: images/<split>/<id>.jpg and masks/<split>/<id>.png with ATR's 18 ids,
mapped to the 15 classes by `label_mapping` in the dataset YAML.

Training writes to <drive-root>/runs_yolo/<run-name>/ (weights/last.pt and best.pt after
every epoch). Run the same command again after a disconnect: it resumes from last.pt.

Example (Colab):
    python scripts/train_yolo_sem.py --drive-root /content/drive/MyDrive/clothseg --model yolo26x-sem.pt \
        --run-name yolo26x_sem_e12 --epochs 12
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clothseg.data.preprocess import DONE_MARKER, cache_name  # noqa: E402
from clothseg.data.splits import load_splits  # noqa: E402
from clothseg.labels import ATR_TO_15, class_names  # noqa: E402


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:  # another file system: copy instead
        shutil.copy2(src, dst)


def build_dataset(cache: Path, out: Path, limit_train: int = 0, limit_val: int = 0) -> Path:
    """Hard-link the cached photos and masks into the YOLO layout and write the dataset YAML."""
    splits = load_splits(cache)
    plan = {"train": splits["train"][: limit_train or None], "val": splits["val"][: limit_val or None]}
    for split, ids in plan.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "masks" / split).mkdir(parents=True, exist_ok=True)
        for idx in ids:
            name = f"{idx:06d}"
            link(cache / "images" / f"{name}.jpg", out / "images" / split / f"{name}.jpg")
            link(cache / "masks" / f"{name}.png", out / "masks" / split / f"{name}.png")
        have = len(list((out / "images" / split).glob("*.jpg")))
        if have != len(ids):
            raise RuntimeError(f"{out / 'images' / split} holds {have} photos, expected {len(ids)}. Delete {out} and run again.")
    names = class_names(15)
    lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val", "masks_dir: masks", "", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    lines += ["", "# ATR's 18 ids -> the project's 15 classes (left/right merged); 255 stays ignored", "label_mapping:"]
    lines += [f"  {atr}: {train}" for atr, train in enumerate(ATR_TO_15)]
    lines += ["  255: ignore_label", ""]
    yaml_path = out / "atr15.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[yolo] dataset ready in {out}: train {len(plan['train'])}, val {len(plan['val'])} photos (hard links to the cache)", flush=True)
    return yaml_path


def finished_epochs(run_dir: Path) -> int:
    results = run_dir / "results.csv"
    if not results.exists():
        return 0
    with open(results, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--local-root", type=Path, default=Path("/content/data"))
    p.add_argument("--height", type=int, default=576, help="Size of the project's dataset cache.")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--model", default="yolo26x-sem.pt", help="Pretrained YOLO26 semantic model to start from.")
    p.add_argument("--run-name", default="yolo26x_sem_e12")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--imgsz", type=int, default=576)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=0, help="Use only the first N training photos (tests only).")
    p.add_argument("--limit-val", type=int, default=0)
    a = p.parse_args()

    cache = a.local_root / cache_name(a.height, a.width)
    if not (cache / DONE_MARKER).exists():
        sys.exit(f"ERROR: no dataset cache at {cache}. Run scripts/prepare_data.py first.")
    from ultralytics import YOLO

    yaml_path = build_dataset(cache, a.local_root / f"yolo_atr_{a.height}x{a.width}", a.limit_train, a.limit_val)
    project = a.drive_root / "runs_yolo"
    run_dir = project / a.run_name
    last = run_dir / "weights" / "last.pt"
    done = finished_epochs(run_dir)
    if done >= a.epochs:
        print(f"[yolo] {a.run_name} already finished {done} epochs. Best weights: {run_dir / 'weights' / 'best.pt'}", flush=True)
        return
    if last.exists():
        print(f"[yolo] RESUMING {a.run_name} after epoch {done} of {a.epochs} from {last}", flush=True)
        YOLO(str(last)).train(resume=True)
    else:
        print(f"[yolo] FRESH START {a.run_name}: {a.model}, {a.epochs} epochs, imgsz {a.imgsz}, batch {a.batch}, seed {a.seed}", flush=True)
        YOLO(a.model).train(data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
                            seed=a.seed, project=str(project), name=a.run_name, exist_ok=True, plots=False)
    print(f"[yolo] FINISHED. Best weights: {run_dir / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
