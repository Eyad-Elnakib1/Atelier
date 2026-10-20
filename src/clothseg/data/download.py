"""Download the raw ATR parquet files once and keep a copy on Google Drive."""
from __future__ import annotations

import shutil
from pathlib import Path

from .storage import atomic_copy, human_size

ATR_REPO_ID = "mattmdjaga/human_parsing_dataset"
_MARKER = "COMPLETE"


def _parquets(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.parquet")) if folder.is_dir() else []


def ensure_raw_atr(drive_raw: Path, local_raw: Path) -> list[Path]:
    """Return local paths of the raw ATR parquet files, fetching them if needed.

    Order of preference:
      1. already on the local disk (same session);
      2. on Drive from an earlier session -> copy to the local disk;
      3. download from the Hugging Face Hub -> copy to Drive for next time.

    A ``COMPLETE`` marker file is written last, so an interrupted copy or download
    is never treated as a full dataset.
    """
    drive_raw, local_raw = Path(drive_raw), Path(local_raw)

    if (local_raw / _MARKER).exists() and _parquets(local_raw):
        files = _parquets(local_raw)
        print(f"[download] raw ATR already on local disk: {len(files)} parquet file(s)", flush=True)
        return files

    local_raw.mkdir(parents=True, exist_ok=True)

    if (drive_raw / _MARKER).exists() and _parquets(drive_raw):
        print(f"[download] copying raw ATR from Drive: {drive_raw}", flush=True)
        for src in _parquets(drive_raw):
            atomic_copy(src, local_raw / src.name)
        (local_raw / _MARKER).touch()
        return _parquets(local_raw)

    print(f"[download] downloading '{ATR_REPO_ID}' from the Hugging Face Hub (about 800 MB)...", flush=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface_hub") from exc
    snap = local_raw / "_hf_snapshot"
    try:
        snapshot_download(repo_id=ATR_REPO_ID, repo_type="dataset", local_dir=str(snap), allow_patterns=["*.parquet"])
    except Exception as exc:  # noqa: BLE001 - re-raised with a clearer message
        raise RuntimeError(
            f"Download of '{ATR_REPO_ID}' failed: {type(exc).__name__}: {exc}\n"
            "Check the internet connection and that the dataset page still exists: "
            f"https://huggingface.co/datasets/{ATR_REPO_ID}"
        ) from exc
    found = _parquets(snap)
    if not found:
        raise RuntimeError(f"Download finished but no .parquet files were found under {snap}.")
    for src in found:
        shutil.move(str(src), str(local_raw / src.name))
    shutil.rmtree(snap, ignore_errors=True)
    (local_raw / _MARKER).touch()
    files = _parquets(local_raw)
    print(f"[download] downloaded {len(files)} file(s), {human_size(sum(f.stat().st_size for f in files))}", flush=True)

    print(f"[download] saving a copy to Drive: {drive_raw}", flush=True)
    for src in files:
        atomic_copy(src, drive_raw / src.name)
    (drive_raw / _MARKER).touch()
    return files
