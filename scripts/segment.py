"""Segment the clothes in one photo or in every photo of a folder.

For each input image it writes, into --out-dir:
  <name>_mask.png      label mask at the photo's own resolution (pixel value = class id)
  <name>_overlay.jpg   the photo with class colours blended on top
  <name>_<garment>.png one transparent cut-out per garment found
  <name>.json          class list and the share of the photo each class covers

Examples:
    python scripts/segment.py --checkpoint best.pt --input photo.jpg --out-dir out
    python scripts/segment.py --checkpoint best.pt --input my_photos/ --out-dir out
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clothseg.inference import EnsembleSegmenter, Segmenter, load_image, save_outputs  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt (or any checkpoint) written by scripts/train.py.")
    p.add_argument("--input", type=Path, required=True, help="An image file or a folder of images.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-area", type=float, default=0.002, help="Skip cut-outs smaller than this fraction of the photo.")
    p.add_argument("--device", default=None, help="cuda or cpu. Default: cuda if available.")
    p.add_argument("--flip-tta", action="store_true", help="Average the prediction with the one on the mirrored photo (slower, slightly more accurate).")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0], help="Input scales to average over, e.g. 0.75 1.0 1.25 (multi-scale test-time augmentation).")
    p.add_argument("--ensemble-with", type=Path, nargs="+", default=[], help="More checkpoints whose predictions are averaged with --checkpoint (same classes; architectures may differ).")
    args = p.parse_args()

    if args.input.is_dir():
        files = sorted(f for f in args.input.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    elif args.input.is_file():
        files = [args.input]
    else:
        sys.exit(f"ERROR: {args.input} does not exist.")
    if not files:
        sys.exit(f"ERROR: no images ({', '.join(sorted(IMAGE_SUFFIXES))}) found in {args.input}.")

    try:
        seg = Segmenter(args.checkpoint, device=args.device, flip_tta=args.flip_tta, scales=tuple(args.scales))
        if args.ensemble_with:
            seg = EnsembleSegmenter([seg] + [Segmenter(c, device=args.device, flip_tta=args.flip_tta, scales=tuple(args.scales))
                                             for c in args.ensemble_with])
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"[segment] model {seg.cfg.model}, {seg.num_classes} classes, input {seg.cfg.height}x{seg.cfg.width}, device {seg.device}", flush=True)

    failed = 0
    for k, path in enumerate(files, 1):
        try:
            image = load_image(path)
            start = time.time()
            mask = seg.predict(image)
            ms = 1000 * (time.time() - start)
            summary = save_outputs(image, mask, seg, args.out_dir, path.stem, args.min_area)
            found = ", ".join(f"{n} {v:.1f}%" for n, v in summary["area_percent"].items() if n != "background")
            print(f"[segment] {k}/{len(files)} {path.name} ({image.shape[1]}x{image.shape[0]}) {ms:.0f} ms | {found or 'nothing found'} | "
                  f"cut-outs: {', '.join(summary['cutouts']) or 'none'}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop a folder run
            failed += 1
            print(f"[segment] {k}/{len(files)} {path.name}: FAILED ({type(exc).__name__}: {exc})", flush=True)
    print(f"[segment] done: {len(files) - failed} ok, {failed} failed. Results in {args.out_dir}", flush=True)
    if failed == len(files):
        sys.exit(1)


if __name__ == "__main__":
    main()
