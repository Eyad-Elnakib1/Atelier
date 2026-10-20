"""Assemble the Hugging Face Space for the demo API into demo/space_build/.

The folder holds everything the Space needs: app.py, Dockerfile, requirements.txt,
README.md (the Space card), the clothseg package and best.pt. Upload the folder's
contents to a new Space (Docker SDK) on huggingface.co, or push it with git.

Run from the project root:
    py -3 tools/build_space.py                 # copies best.pt from the project root
    py -3 tools/build_space.py --no-weights    # without best.pt (set MODEL_REPO in the Space instead)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "demo" / "server"
OUT = ROOT / "demo" / "space_build"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-weights", action="store_true", help="Do not copy best.pt.")
    a = p.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "src").mkdir(parents=True)
    for name in ("app.py", "Dockerfile", "requirements.txt"):
        shutil.copy2(SERVER / name, OUT / name)
    shutil.copy2(SERVER / "SPACE_README.md", OUT / "README.md")
    shutil.copytree(ROOT / "src" / "clothseg", OUT / "src" / "clothseg", ignore=shutil.ignore_patterns("__pycache__"))
    if not a.no_weights:
        weights = ROOT / "best.pt"
        if not weights.is_file():
            raise SystemExit(f"ERROR: {weights} not found. Put the final model there, or use --no-weights.")
        shutil.copy2(weights, OUT / "best.pt")
    files = sorted(str(f.relative_to(OUT)).replace("\\", "/") for f in OUT.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1024**2
    print(f"{OUT} ready: {len(files)} files, {size:.0f} MB")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
