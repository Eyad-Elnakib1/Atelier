"""Evaluate a checkpoint on the validation or test split at the original image resolution.

Predictions come from the same ``Segmenter.predict`` that the inference pipeline
uses, and are compared with the full-size ground-truth masks (``masks_orig``).

Writes into --out-dir:
  metrics.json            overall numbers
  per_class.csv           IoU, Dice, accuracy and Boundary IoU per class
  confusion_matrix.csv/.png
  best_examples.png, worst_examples.png

Use --split val while choosing models. Run --split test once, for the final model only.

Example (Colab):
    python scripts/evaluate.py --checkpoint /content/drive/MyDrive/clothseg/runs/<run>/checkpoints/best.pt \
        --split val --out-dir /content/drive/MyDrive/clothseg/reports/<run>_val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clothseg.data.preprocess import DONE_MARKER, cache_name  # noqa: E402
from clothseg.data.splits import load_splits  # noqa: E402
from clothseg.inference import EnsembleSegmenter, Segmenter  # noqa: E402
from clothseg.labels import ATR_TO_15, build_lut, garment_ids  # noqa: E402
from clothseg.metrics import ConfusionMatrix  # noqa: E402


def boundary(mask: np.ndarray, d: int) -> np.ndarray:
    """Pixels of a binary mask within ``d`` pixels of its contour (Boundary IoU, Cheng et al. 2021)."""
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=d)[1:-1, 1:-1]
    return mask & ~eroded.astype(bool)


def examples_figure(seg: Segmenter, cache: Path, items: list[tuple[float, int]], lut: np.ndarray, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(items), 3, figsize=(7.5, 3.6 * len(items)), squeeze=False)
    for row, (score, idx) in zip(axes, items):
        image = np.asarray(Image.open(cache / "images" / f"{idx:06d}.jpg").convert("RGB"))
        gt = lut[np.asarray(Image.open(cache / "masks_orig" / f"{idx:06d}.png"))]
        pred = seg.predict(image)
        gt_col = seg.colorize(np.where(gt == 255, 0, gt))
        for ax, pic, name in zip(row, (image, gt_col, seg.colorize(pred)), (f"#{idx}  mIoU {score:.2f}", "ground truth", "prediction")):
            ax.imshow(pic)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    # The heading is placed above the laid-out grid so it cannot overlap the row titles.
    fig.suptitle(title, fontsize=12, y=1.0, va="bottom")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--local-root", type=Path, default=Path("/content/data"), help="Folder that holds the extracted dataset cache.")
    p.add_argument("--num-examples", type=int, default=6, help="Images in each of the best/worst figures.")
    p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images (tests only).")
    p.add_argument("--flip-tta", action="store_true", help="Average the prediction with the one on the mirrored photo (about twice the time).")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0], help="Input scales to average over, e.g. 0.75 1.0 1.25 (multi-scale test-time augmentation).")
    p.add_argument("--ensemble-with", type=Path, nargs="+", default=[], help="More checkpoints whose predictions are averaged with --checkpoint (same classes; architectures may differ).")
    args = p.parse_args()

    try:
        seg = Segmenter(args.checkpoint, flip_tta=args.flip_tta, scales=tuple(args.scales))
        if args.ensemble_with:
            seg = EnsembleSegmenter([seg] + [Segmenter(c, flip_tta=args.flip_tta, scales=tuple(args.scales)) for c in args.ensemble_with])
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    cache = args.local_root / cache_name(seg.cfg.height, seg.cfg.width)
    if not (cache / DONE_MARKER).exists():
        sys.exit(f"ERROR: dataset cache not found at {cache}. Run scripts/prepare_data.py first.")
    ids = load_splits(cache)[args.split]
    if args.limit:
        ids = ids[:args.limit]
    if args.split == "test":
        print("[eval] NOTE: this is the TEST split. Use it once, for the final model; choose models on --split val.", flush=True)
    print(f"[eval] {seg.cfg.model} (epoch {seg.epoch}) | {seg.num_classes} classes | {len(ids)} {args.split} images | original resolution | {seg.device}", flush=True)

    nc, names = seg.num_classes, seg.names
    lut = build_lut(nc)
    cm = ConfusionMatrix(nc, seg.device)
    to15 = np.array(ATR_TO_15 + (255,) * (256 - len(ATR_TO_15)), dtype=np.uint8) if nc == 18 else None
    cm15 = ConfusionMatrix(15, seg.device) if nc == 18 else None
    b_inter, b_union = np.zeros(nc), np.zeros(nc)
    per_image, seconds = [], 0.0
    start, last = time.time(), time.time()
    for k, idx in enumerate(ids, 1):
        image = np.asarray(Image.open(cache / "images" / f"{idx:06d}.jpg").convert("RGB"))
        gt = lut[np.asarray(Image.open(cache / "masks_orig" / f"{idx:06d}.png"))]
        t0 = time.time()
        # The cached photo was resized for training; predict() scales its output to any size,
        # so the mask is requested at the ground truth's (original) resolution.
        full = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) if image.shape[:2] != gt.shape else image
        pred = seg.predict(full)
        seconds += time.time() - t0
        cm.update(torch.from_numpy(pred).to(seg.device), torch.from_numpy(gt).to(seg.device))
        if cm15 is not None:
            gt15 = np.where(gt == 255, 255, to15[gt])
            cm15.update(torch.from_numpy(to15[pred]).to(seg.device), torch.from_numpy(gt15).to(seg.device))

        valid = gt != 255
        d = max(1, round(0.02 * float(np.hypot(*gt.shape))))
        ious = []
        for c in np.union1d(np.unique(gt[valid]), np.unique(pred[valid])):
            g, q = (gt == c) & valid, (pred == c) & valid
            ious.append((g & q).sum() / max(1, (g | q).sum()))
            bg, bq = boundary(g, d), boundary(q, d)
            b_inter[c] += (bg & bq).sum()
            b_union[c] += (bg | bq).sum()
        per_image.append((float(np.mean(ious)), idx))
        if time.time() - last > 20:
            print(f"[eval] {k}/{len(ids)}  ({k / (time.time() - start):.1f} img/s)", flush=True)
            last = time.time()

    m = cm.compute()
    b_iou = np.where(b_union > 0, b_inter / np.maximum(b_union, 1), np.nan)
    matrix = cm.matrix.cpu().numpy()
    gt_px = matrix.sum(1)
    acc = np.where(gt_px > 0, np.diag(matrix) / np.maximum(gt_px, 1), np.nan)
    garment = garment_ids(nc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'id':>3} {'class':<14}{'IoU':>8}{'Dice':>8}{'acc':>8}{'bound.IoU':>11}{'GT pixel %':>12}")
    with open(args.out_dir / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "iou", "dice", "accuracy", "boundary_iou", "gt_pixel_percent", "is_garment"])
        for i, n in enumerate(names):
            row = [m["iou"][n], m["dice"][n], acc[i], b_iou[i], 100 * gt_px[i] / max(1, gt_px.sum())]
            print(f"{i:>3} {n:<14}" + "".join(f"{v:>8.4f}" for v in row[:3]) + f"{row[3]:>11.4f}{row[4]:>12.3f}")
            w.writerow([i, n] + [round(float(v), 5) for v in row] + [int(i in garment)])

    summary = {
        "checkpoint": str(args.checkpoint), "ensemble_with": [str(c) for c in args.ensemble_with], "model": seg.cfg.model, "epoch": seg.epoch, "split": args.split,
        "num_images": len(ids), "num_classes": nc, "resolution": "original", "flip_tta": args.flip_tta, "scales": list(args.scales),
        "miou": m["miou"], "garment_miou": m["garment_miou"], "clothing_binary_iou": m["clothing_binary_iou"],
        "pixel_acc": m["pixel_acc"], "mean_class_acc": m["mean_class_acc"], "mean_dice": m["mean_dice"],
        "boundary_miou": float(np.nanmean(b_iou)), "garment_boundary_miou": float(np.nanmean(b_iou[garment])),
        "core_garment_miou": float(np.mean([m["iou"][n] for n in ("upper-clothes", "pants", "skirt", "dress")])),
        "ms_per_image": 1000 * seconds / max(1, len(ids)),
    }
    if cm15 is not None:
        m15 = cm15.compute()
        summary.update({"miou_as_15_classes": m15["miou"], "garment_miou_as_15_classes": m15["garment_miou"]})
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "\n".join(f"{k:<28}{v:.4f}" if isinstance(v, float) else f"{k:<28}{v}" for k, v in summary.items()))

    np.savetxt(args.out_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",", header=",".join(names), comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(nc), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(nc), names, fontsize=8)
    for i in range(nc):
        for j in range(nc):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Confusion matrix, {args.split} split (% of each ground-truth class)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    per_image.sort()
    n = min(args.num_examples, len(per_image))
    examples_figure(seg, cache, per_image[:n], lut, f"Worst {n} predictions ({args.split})", args.out_dir / "worst_examples.png")
    examples_figure(seg, cache, per_image[-n:][::-1], lut, f"Best {n} predictions ({args.split})", args.out_dir / "best_examples.png")
    print(f"\n[eval] saved metrics.json, per_class.csv, confusion_matrix.csv/.png, best_examples.png, worst_examples.png in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
