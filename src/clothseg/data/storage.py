"""Safe file transfer between Google Drive and the local Colab disk.

Drive is slow for many small files and a session can die at any moment, so:
  * datasets travel as one tar archive, never as thousands of files;
  * every write to Drive goes to a temporary name first and is renamed at the end,
    so a half-written file is never mistaken for a complete one.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import time
from pathlib import Path


def human_size(num_bytes: float) -> str:
    """Format a byte count, e.g. ``1.5 GB``."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def atomic_copy(src: Path, dst: Path, quiet: bool = False) -> None:
    """Copy ``src`` to ``dst`` so that ``dst`` only ever exists as a complete file."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    start = time.time()
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Copy of {src} to {dst} is incomplete (size mismatch). Check Drive free space and retry.")
    os.replace(tmp, dst)
    if not quiet:
        print(f"[storage] copied {src.name} ({human_size(dst.stat().st_size)}) -> {dst.parent} in {time.time() - start:.0f}s", flush=True)


def make_tar(src_dir: Path, tar_path: Path) -> None:
    """Pack ``src_dir`` into an uncompressed tar (images are already compressed)."""
    src_dir, tar_path = Path(src_dir), Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_name(tar_path.name + ".tmp")
    start = time.time()
    with tarfile.open(tmp, "w") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    os.replace(tmp, tar_path)
    print(f"[storage] packed {src_dir.name} -> {tar_path.name} ({human_size(tar_path.stat().st_size)}) in {time.time() - start:.0f}s", flush=True)


def extract_tar(tar_path: Path, dest_parent: Path) -> None:
    """Extract a tar created by :func:`make_tar` into ``dest_parent``."""
    tar_path, dest_parent = Path(tar_path), Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest_parent, filter="data")
    print(f"[storage] extracted {tar_path.name} -> {dest_parent} in {time.time() - start:.0f}s", flush=True)
