"""Turn the raw ATR parquet files into a training cache of small image files.

For every sample the cache holds:
  * ``images/<idx>.jpg``      image resized to fit inside the training resolution
  * ``masks/<idx>.png``       label mask at the same size (original 18 ATR ids)
  * ``masks_orig/<idx>.png``  label mask at the original size, for evaluation

plus ``meta.csv`` (one row per sample), ``splits.json`` (fixed-seed train/val/test)
and ``stats.json`` (summary). Images keep their aspect ratio; padding to the exact
training size happens at load time.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from ..labels import ATR_CLASSES

NUM_ATR = len(ATR_CLASSES)
DONE_MARKER = "DONE"
META_FIELDS = ["idx", "orig_w", "orig_h", "w", "h", "md5", "dhash", "bad_px"] + [f"c{i}" for i in range(NUM_ATR)]


def cache_name(height: int, width: int) -> str:
    """Directory/archive name of the cache for a given training resolution."""
    return f"atr_{height}x{width}"


def _dhash(img: Image.Image) -> str:
    """64-bit difference hash; identical values indicate near-duplicate images."""
    small = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def _fit_size(w: int, h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Largest (w, h) with the same aspect ratio that fits inside (max_w, max_h)."""
    scale = min(max_w / w, max_h / h)
    return max(1, round(w * scale)), max(1, round(h * scale))


def _process_one(task: tuple) -> dict:
    """Decode, validate, resize and save one sample. Runs in a worker process."""
    idx, img_bytes, mask_bytes, out_dir, max_h, max_w = task
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask = np.asarray(mask_img)
        if mask.ndim == 3:
            # Some exports store a label map as 3 equal channels.
            if not (mask[..., 0] == mask[..., 1]).all():
                return {"idx": idx, "error": f"mask has {mask.shape[2]} different channels (mode {mask_img.mode})"}
            mask = mask[..., 0]
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        w, h = img.size
        if mask.shape != (h, w):
            return {"idx": idx, "error": f"image is {w}x{h} but mask is {mask.shape[1]}x{mask.shape[0]}"}

        hist = np.bincount(mask.ravel(), minlength=256)
        new_w, new_h = _fit_size(w, h, max_w, max_h)
        name = f"{idx:06d}"
        img.resize((new_w, new_h), Image.LANCZOS).save(Path(out_dir, "images", name + ".jpg"), quality=95)
        mask_pil = Image.fromarray(mask)
        mask_pil.resize((new_w, new_h), Image.NEAREST).save(Path(out_dir, "masks", name + ".png"))
        mask_pil.save(Path(out_dir, "masks_orig", name + ".png"))

        row = {
            "idx": idx, "orig_w": w, "orig_h": h, "w": new_w, "h": new_h,
            "md5": hashlib.md5(img_bytes).hexdigest(), "dhash": _dhash(img),
            "bad_px": int(hist[NUM_ATR:].sum()),
        }
        row.update({f"c{i}": int(hist[i]) for i in range(NUM_ATR)})
        return row
    except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the run
        return {"idx": idx, "error": f"{type(exc).__name__}: {exc}"}


def _find_columns(schema: pa.Schema) -> tuple[str, str]:
    """Locate the image and mask columns in a Hugging Face image parquet file."""
    structs = [f.name for f in schema if pa.types.is_struct(f.type) and f.type.get_field_index("bytes") >= 0]
    if "image" in structs and "mask" in structs:
        return "image", "mask"
    masks = [n for n in structs if any(k in n.lower() for k in ("mask", "label", "annot", "seg"))]
    images = [n for n in structs if n not in masks]
    if len(masks) == 1 and len(images) == 1:
        return images[0], masks[0]
    raise RuntimeError(
        "Could not identify the image and mask columns in the parquet file.\n"
        f"Columns with image bytes: {structs}\nFull schema:\n{schema}"
    )


def _make_splits(rows: list[dict], seed: int, val_size: int, test_size: int) -> dict:
    """Drop exact duplicate images, then split the rest with a fixed seed."""
    seen: dict[str, int] = {}
    keep, dropped = [], []
    for r in rows:  # rows are sorted by idx, so the first copy is always kept
        if r["md5"] in seen:
            dropped.append(r["idx"])
        else:
            seen[r["md5"]] = r["idx"]
            keep.append(r["idx"])
    n = len(keep)
    if n < 10 * (val_size + test_size):
        # Dataset is much smaller than expected: fall back to proportions.
        val_size, test_size = max(1, int(0.04 * n)), max(1, int(0.06 * n))
        print(f"[preprocess] WARNING: only {n} samples, using val={val_size} test={test_size}", flush=True)
    perm = np.random.default_rng(seed).permutation(np.array(keep, dtype=np.int64))
    return {
        "seed": seed,
        "test": sorted(int(i) for i in perm[:test_size]),
        "val": sorted(int(i) for i in perm[test_size:test_size + val_size]),
        "train": sorted(int(i) for i in perm[test_size + val_size:]),
        "dropped_exact_duplicates": dropped,
    }


def build_cache(parquet_files: list[Path], out_dir: Path, height: int, width: int,
                seed: int = 42, val_size: int = 706, test_size: int = 1000, workers: int = 2) -> Path:
    """Build the preprocessed cache in ``out_dir`` and return its path.

    Work is done in ``<out_dir>.partial`` and renamed at the end, so an interrupted
    run never leaves something that looks like a finished cache.
    """
    out_dir = Path(out_dir)
    partial = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    for sub in ("images", "masks", "masks_orig"):
        (partial / sub).mkdir(parents=True, exist_ok=True)

    total = sum(pq.ParquetFile(f).metadata.num_rows for f in parquet_files)
    print(f"[preprocess] {total} samples in {len(parquet_files)} parquet file(s); target fits inside {height}x{width} (HxW)", flush=True)

    rows, errors = [], []
    idx, start, last_print = 0, time.time(), time.time()
    with Pool(workers) as pool:
        for path in sorted(parquet_files):
            pf = pq.ParquetFile(path)
            img_col, mask_col = _find_columns(pf.schema_arrow)
            for batch in pf.iter_batches(batch_size=64, columns=[img_col, mask_col]):
                images = batch.column(img_col).field("bytes").to_pylist()
                masks = batch.column(mask_col).field("bytes").to_pylist()
                tasks = [(idx + k, images[k], masks[k], str(partial), height, width) for k in range(len(images))]
                idx += len(tasks)
                for res in pool.map(_process_one, tasks):
                    (errors if "error" in res else rows).append(res)
                if time.time() - last_print > 20:
                    rate = idx / (time.time() - start)
                    print(f"[preprocess] {idx}/{total} ({100 * idx / total:.0f}%)  {rate:.0f} img/s  ETA {(total - idx) / rate:.0f}s", flush=True)
                    last_print = time.time()
    print(f"[preprocess] processed {idx} samples in {time.time() - start:.0f}s; {len(errors)} failed", flush=True)

    for e in errors[:20]:
        print(f"[preprocess]   sample {e['idx']}: {e['error']}", flush=True)
    if len(errors) > 0.01 * max(1, idx):
        raise RuntimeError(f"{len(errors)} of {idx} samples failed (more than 1%). See the messages above; the cache was not created.")

    rows.sort(key=lambda r: r["idx"])
    with open(partial / "meta.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    splits = _make_splits(rows, seed, val_size, test_size)
    splits["failed_samples"] = [e["idx"] for e in errors]
    with open(partial / "splits.json", "w") as f:
        json.dump(splits, f)

    stats = {
        "height": height, "width": width, "seed": seed,
        "num_samples": len(rows), "num_failed": len(errors),
        "num_exact_duplicates_dropped": len(splits["dropped_exact_duplicates"]),
        "split_sizes": {k: len(splits[k]) for k in ("train", "val", "test")},
        "samples_with_invalid_label_values": sum(1 for r in rows if r["bad_px"] > 0),
    }
    with open(partial / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    (partial / DONE_MARKER).touch()

    shutil.rmtree(out_dir, ignore_errors=True)
    partial.rename(out_dir)
    print(f"[preprocess] cache ready: {out_dir}\n[preprocess] {json.dumps(stats)}", flush=True)
    return out_dir
