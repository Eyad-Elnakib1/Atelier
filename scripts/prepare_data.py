"""Prepare the ATR dataset for training. Safe to run at the start of every session.

What it does, skipping every step that is already done:
  1. If the preprocessed cache is already on the local disk -> nothing to do.
  2. Else, if a cache archive exists on Drive -> copy it to the local disk and extract.
  3. Else build it: get the raw ATR parquet files (Drive copy, or download from the
     Hugging Face Hub and save to Drive), resize, validate, split with a fixed seed,
     then pack the cache into one tar file and save it to Drive.

In every case it finishes by moving near-duplicate photos that sit in more than one
split into train (see clothseg/data/splits.py), so val and test hold no copy of a
training image. The step is deterministic, so every session gets the same splits.

Example (Colab):
    python scripts/prepare_data.py --drive-root /content/drive/MyDrive/clothseg
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clothseg.data.download import ensure_raw_atr  # noqa: E402
from clothseg.data.preprocess import DONE_MARKER, build_cache, cache_name  # noqa: E402
from clothseg.data.splits import move_cross_split_near_duplicates  # noqa: E402
from clothseg.data.storage import atomic_copy, extract_tar, human_size, make_tar  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-root", type=Path, required=True, help="Project folder on Google Drive (persistent).")
    p.add_argument("--local-root", type=Path, default=Path("/content/data"), help="Fast local folder (lost when the session ends).")
    p.add_argument("--height", type=int, default=576, help="Training height in pixels.")
    p.add_argument("--width", type=int, default=384, help="Training width in pixels.")
    p.add_argument("--seed", type=int, default=42, help="Seed of the train/val/test split.")
    p.add_argument("--val-size", type=int, default=706)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2, help="Worker processes (Colab has 2 CPU cores).")
    p.add_argument("--min-free-gb", type=float, default=6.0, help="Stop if the local disk has less free space than this.")
    p.add_argument("--keep-cross-split-duplicates", action="store_true", help="Do not move cross-split near-duplicates into train.")
    p.add_argument("--force-rebuild", action="store_true", help="Ignore existing caches and rebuild from the raw files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    local_cache = prepare(args)
    if not args.keep_cross_split_duplicates:
        move_cross_split_near_duplicates(local_cache)


def prepare(args: argparse.Namespace) -> Path:
    """Make the cache available on the local disk and return its path."""
    start = time.time()
    name = cache_name(args.height, args.width)
    local_cache = args.local_root / name
    drive_tar = args.drive_root / "data" / "cache" / f"{name}.tar"
    args.local_root.mkdir(parents=True, exist_ok=True)

    if not args.drive_root.parent.exists():
        sys.exit(f"ERROR: {args.drive_root.parent} does not exist. Mount Google Drive first (run the 'Mount Drive' cell).")
    free = shutil.disk_usage(args.local_root).free
    print(f"[prepare] cache name: {name} | local: {local_cache} | Drive archive: {drive_tar}", flush=True)
    print(f"[prepare] free local disk: {human_size(free)}", flush=True)
    if free < args.min_free_gb * 1024**3:
        sys.exit(f"ERROR: less than {args.min_free_gb:g} GB free on the local disk. Free some space or restart the runtime.")

    if args.force_rebuild:
        print("[prepare] --force-rebuild: ignoring existing caches", flush=True)
        shutil.rmtree(local_cache, ignore_errors=True)
    elif (local_cache / DONE_MARKER).exists():
        print("[prepare] cache already on the local disk. Nothing to do.", flush=True)
        return local_cache
    elif drive_tar.exists():
        print(f"[prepare] found archive on Drive ({human_size(drive_tar.stat().st_size)}). Copying and extracting...", flush=True)
        local_tar = args.local_root / drive_tar.name
        atomic_copy(drive_tar, local_tar)
        shutil.rmtree(local_cache, ignore_errors=True)
        extract_tar(local_tar, args.local_root)
        local_tar.unlink()
        if not (local_cache / DONE_MARKER).exists():
            sys.exit(f"ERROR: the archive {drive_tar} did not contain a complete cache. Delete it on Drive and rerun to rebuild.")
        print(f"[prepare] done in {time.time() - start:.0f}s", flush=True)
        return local_cache

    print("[prepare] no cache found. Building it from the raw dataset (first run only).", flush=True)
    parquet_files = ensure_raw_atr(args.drive_root / "data" / "raw" / "atr", args.local_root / "raw" / "atr")
    build_cache(parquet_files, local_cache, args.height, args.width, seed=args.seed,
                val_size=args.val_size, test_size=args.test_size, workers=args.workers)

    local_tar = args.local_root / drive_tar.name
    make_tar(local_cache, local_tar)
    atomic_copy(local_tar, drive_tar)
    local_tar.unlink()
    print(f"[prepare] all done in {time.time() - start:.0f}s. Archive saved to {drive_tar}", flush=True)
    return local_cache


if __name__ == "__main__":
    main()
