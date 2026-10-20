"""Check that the preprocessed ATR cache is healthy before spending GPU time on it.

Prints:
  * split sizes, dropped duplicates, failed samples
  * original image size distribution (decides whether the training resolution fits)
  * per-class pixel frequencies for the 18 ATR classes and the merged 15 classes
  * possible near-duplicate images that sit in different splits (leakage)
  * a spot check that saved images and masks open and match
  * which side of the image "left-arm" is on (needed for flip augmentation with 18 classes)
and saves a figure of random samples with mask overlays.

Example (Colab):
    python scripts/sanity_check.py --out-dir /content/drive/MyDrive/clothseg/reports
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clothseg.labels import ATR_CLASSES, ATR_TO_15, CLASSES_15, build_lut, class_names, palette  # noqa: E402
from clothseg.data.preprocess import DONE_MARKER, cache_name  # noqa: E402

NUM_ATR = len(ATR_CLASSES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local-root", type=Path, default=Path("/content/data"))
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--out-dir", type=Path, required=True, help="Where to save the overlay figure and the report (Drive).")
    p.add_argument("--num-samples", type=int, default=12, help="Samples in the overlay figure.")
    p.add_argument("--num-classes", type=int, default=15, choices=(15, 18), help="Label space used for the overlay figure.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pct(values, qs=(0, 5, 25, 50, 75, 95, 100)) -> str:
    return "  ".join(f"p{q}={np.percentile(values, q):.0f}" for q in qs)


def class_table(title: str, names, pixels: np.ndarray, images: np.ndarray, n_images: int) -> None:
    print(f"\n{title}")
    print(f"{'id':>3} {'class':<14} {'pixel %':>9} {'% of person':>12} {'images':>7} {'image %':>8}")
    total, fg = pixels.sum(), pixels[1:].sum()
    for i, name in enumerate(names):
        person = f"{100 * pixels[i] / fg:>12.2f}" if i else f"{'-':>12}"
        print(f"{i:>3} {name:<14} {100 * pixels[i] / total:>9.3f} {person} {int(images[i]):>7} {100 * images[i] / n_images:>8.1f}")


def make_overlay_figure(cache: Path, ids: list[int], num_classes: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lut, pal, names = build_lut(num_classes), palette(num_classes), class_names(num_classes)
    cols = 4
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.4, rows * 3.4))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, idx in enumerate(ids):
        img = np.asarray(Image.open(cache / "images" / f"{idx:06d}.jpg").convert("RGB"))
        mask = lut[np.asarray(Image.open(cache / "masks" / f"{idx:06d}.png"))]
        colour = pal[np.clip(mask, 0, num_classes - 1)]
        overlay = (0.4 * img + 0.6 * colour).astype(np.uint8)
        overlay[mask == 0] = (0.5 * img[mask == 0]).astype(np.uint8)
        r, c = divmod(k, cols)
        axes[r, 2 * c].imshow(img)
        axes[r, 2 * c].set_title(f"#{idx}  {img.shape[1]}x{img.shape[0]}", fontsize=8)
        axes[r, 2 * c + 1].imshow(overlay)
    handles = [Patch(color=pal[i] / 255.0, label=names[i]) for i in range(1, num_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, num_classes - 1), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache = args.local_root / cache_name(args.height, args.width)
    if not (cache / DONE_MARKER).exists():
        sys.exit(f"ERROR: no finished cache at {cache}. Run scripts/prepare_data.py first.")
    problems: list[str] = []

    with open(cache / "meta.csv", newline="") as f:
        meta = {int(r["idx"]): r for r in csv.DictReader(f)}
    splits = json.loads((cache / "splits.json").read_text())
    stats = json.loads((cache / "stats.json").read_text())
    split_of = {i: s for s in ("train", "val", "test") for i in splits[s]}

    print("=" * 70 + "\nDATASET SUMMARY\n" + "=" * 70)
    print(f"cache: {cache}")
    print(f"samples processed: {stats['num_samples']} | failed to decode: {stats['num_failed']} | "
          f"exact duplicates dropped: {stats['num_exact_duplicates_dropped']}")
    print(f"split sizes (seed {splits['seed']}): " + ", ".join(f"{s}={len(splits[s])}" for s in ("train", "val", "test")))
    overlap = (set(splits["train"]) & set(splits["val"])) | (set(splits["train"]) & set(splits["test"])) | (set(splits["val"]) & set(splits["test"]))
    if overlap:
        problems.append(f"{len(overlap)} sample ids are in more than one split")
    if stats["samples_with_invalid_label_values"]:
        problems.append(f"{stats['samples_with_invalid_label_values']} masks contain label values above {NUM_ATR - 1} (they will be ignored in training)")
    if stats["num_failed"]:
        problems.append(f"{stats['num_failed']} samples could not be decoded and were skipped")

    used = [meta[i] for i in split_of]
    ow = np.array([int(r["orig_w"]) for r in used])
    oh = np.array([int(r["orig_h"]) for r in used])
    ratio = oh / ow
    print("\nORIGINAL IMAGE SIZES")
    print(f"width : {pct(ow)}\nheight: {pct(oh)}")
    print(f"portrait (H/W > 1.1): {100 * (ratio > 1.1).mean():.1f}% | square: {100 * ((ratio >= 0.9) & (ratio <= 1.1)).mean():.1f}% | "
          f"landscape: {100 * (ratio < 0.9).mean():.1f}%")
    print("most common WxH: " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in Counter(zip(ow.tolist(), oh.tolist())).most_common(5)))
    upscaled = float(((oh < args.height) & (ow < args.width)).mean())
    print(f"images smaller than the training size {args.width}x{args.height} (they are upscaled): {100 * upscaled:.1f}%")
    if upscaled > 0.5:
        problems.append(f"{100 * upscaled:.0f}% of images are smaller than the training resolution; a higher resolution will not add detail")

    lut15 = np.array(ATR_TO_15)
    for split in ("train", "val", "test"):
        rows = [meta[i] for i in splits[split]]
        counts = np.array([[int(r[f"c{c}"]) for c in range(NUM_ATR)] for r in rows], dtype=np.int64)
        class_table(f"CLASS FREQUENCIES, {split.upper()} ({len(rows)} images), 18 ATR classes", ATR_CLASSES, counts.sum(0), (counts > 0).sum(0), len(rows))
        counts15 = np.zeros((len(rows), 15), dtype=np.int64)
        np.add.at(counts15, (slice(None), lut15), counts)
        class_table(f"CLASS FREQUENCIES, {split.upper()}, merged 15 classes", CLASSES_15, counts15.sum(0), (counts15 > 0).sum(0), len(rows))
        empty = int((counts[:, 1:].sum(1) == 0).sum())
        if empty:
            problems.append(f"{empty} {split} masks are entirely background")
        missing = [CLASSES_15[c] for c in range(15) if (counts15[:, c] > 0).sum() == 0]
        if missing:
            problems.append(f"classes with no example in {split}: {missing}")

    groups = defaultdict(list)
    for i in split_of:
        groups[meta[i]["dhash"]].append(i)
    cross = [g for g in groups.values() if len({split_of[i] for i in g}) > 1]
    print(f"\nPOSSIBLE NEAR-DUPLICATES ACROSS SPLITS (same perceptual hash): {len(cross)} groups, {sum(len(g) for g in cross)} images")
    for g in cross[:5]:
        print("   " + ", ".join(f"#{i}({split_of[i]})" for i in g))
    if len(cross) > 0.01 * len(split_of):
        problems.append(f"{len(cross)} possible near-duplicate groups span different splits; view a few to confirm")

    rng = np.random.default_rng(args.seed)
    check_ids = rng.choice(np.array(sorted(split_of)), size=min(300, len(split_of)), replace=False)
    bad_files, left_on_right, both_arms = 0, 0, 0
    for i in check_ids:
        try:
            img = Image.open(cache / "images" / f"{i:06d}.jpg")
            m = np.asarray(Image.open(cache / "masks" / f"{i:06d}.png"))
            mo = np.asarray(Image.open(cache / "masks_orig" / f"{i:06d}.png"))
            r = meta[int(i)]
            ok = (img.size == (m.shape[1], m.shape[0]) and m.shape[0] <= args.height and m.shape[1] <= args.width
                  and mo.shape == (int(r["orig_h"]), int(r["orig_w"])) and m.ndim == 2)
            bad_files += not ok
            if (m == 14).any() and (m == 15).any():
                both_arms += 1
                left_on_right += np.where(m == 14)[1].mean() > np.where(m == 15)[1].mean()
        except Exception as exc:  # noqa: BLE001
            bad_files += 1
            print(f"   sample {i}: {type(exc).__name__}: {exc}")
    print(f"\nFILE SPOT CHECK: {len(check_ids)} samples opened, {bad_files} with problems")
    if bad_files:
        problems.append(f"{bad_files} of {len(check_ids)} spot-checked samples have unreadable or mismatched files")
    if both_arms:
        share = 100 * left_on_right / both_arms
        print(f"LEFT/RIGHT CONVENTION: in {share:.0f}% of {both_arms} samples the 'left-arm' label is on the right half of the image "
              f"({'labels follow the person' if share > 65 else 'labels follow the image' if share < 35 else 'no clear convention'}; a mirror flip must swap left/right labels in the 18-class setup)")

    fig_ids = [int(i) for i in rng.choice(np.array(splits["train"]), size=min(args.num_samples, len(splits["train"])), replace=False)]
    fig_path = args.out_dir / f"sanity_overlays_{args.num_classes}cls.png"
    make_overlay_figure(cache, fig_ids, args.num_classes, fig_path)
    print(f"\noverlay figure saved: {fig_path}")

    print("\n" + "=" * 70)
    if problems:
        print(f"RESULT: {len(problems)} thing(s) to look at:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("RESULT: no problems found.")
    print("=" * 70)


if __name__ == "__main__":
    main()
