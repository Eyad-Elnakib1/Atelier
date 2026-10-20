"""Web API for the clothes-segmentation model, used by the demo website.

It wraps the same ``Segmenter`` that produced the report's numbers, so a photo sent here
goes through exactly the evaluated pipeline (including flip averaging, if asked for).
Photos are processed in memory and never written to disk or logged.

Endpoints
    GET  /api/health    model loaded? which device?
    GET  /api/info      classes, colours and the model card (metrics from report_latex/report.pdf)
    POST /api/segment   multipart field "image" -> label mask (PNG, base64) + classes found

Run locally (from the project root):
    py -3 -m uvicorn demo.server.app:app --port 7860
Settings (environment variables):
    MODEL_PATH       path to best.pt (default: best.pt next to this file, then the project root)
    MODEL_REPO       optional Hugging Face model repo to download best.pt from if MODEL_PATH is missing
    MAX_SIDE         long side the photo is reduced to before segmentation (default 1024)
    DEVICE           cpu or cuda (default: cuda if available)
    ALLOWED_ORIGINS  comma-separated origins for CORS (default *)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parents[1] / "src"):  # Space image, or the project checkout
    if (candidate / "clothseg").is_dir():
        sys.path.insert(0, str(candidate))
        break

from clothseg.inference import Segmenter, load_image  # noqa: E402
from clothseg.labels import garment_ids  # noqa: E402

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1024"))

# Figures from report_latex/report.pdf (test split evaluated once; CCP is the outside dataset of section IV-B).
MODEL_CARD = {
    "name": "SegFormer-B3, fine-tuned on ATR",
    "parameters_millions": 47.2,
    "training": "30 epochs on 16,038 ATR photos, 576x384, cross-entropy + Lovasz-Softmax",
    "test": {"miou": 0.789, "garment_miou": 0.744, "clothing_vs_rest_iou": 0.930, "pixel_accuracy": 0.970},
    "outside_dataset_ccp": {"garment_miou": 0.677, "public_model_garment_miou": 0.635},
    "limitations": [
        "One label map for everyone in the photo: people are not separated.",
        "Garments ATR has no class for (gowns, robes, coats) are forced into upper-clothes, pants or dress.",
        "Dress against top plus skirt is the most common mistake.",
        "Thin items (belt, sunglasses, scarf) are the weakest classes.",
        "Flat objects held in the hands can be taken for a bag.",
        "Small or distant people are segmented coarsely or missed.",
    ],
    "licence": "Research and evaluation only (SegFormer weights: NVIDIA licence; ATR has no stated licence).",
}


def _model_path() -> Path:
    env = os.environ.get("MODEL_PATH")
    candidates = [Path(env)] if env else [HERE / "best.pt", HERE.parents[1] / "best.pt"]
    for c in candidates:
        if c.is_file():
            return c
    repo = os.environ.get("MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename="best.pt"))
    raise FileNotFoundError(f"best.pt not found (looked at {', '.join(map(str, candidates))}); set MODEL_PATH or MODEL_REPO.")


torch.set_num_threads(max(1, os.cpu_count() or 1))
SEG = Segmenter(_model_path(), device=os.environ.get("DEVICE") or None)
LOCK = threading.Lock()  # one photo at a time: predictable memory on a small server, and flip_tta is set per request
GARMENTS = set(garment_ids(SEG.num_classes))

app = FastAPI(title="Clothes segmentation API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": SEG.cfg.model, "epoch": SEG.epoch, "device": str(SEG.device)}


@app.get("/api/info")
def info() -> dict:
    classes = [{"id": i, "name": n, "color": SEG.palette[i].tolist(), "garment": i in GARMENTS}
               for i, n in enumerate(SEG.names)]
    return {"classes": classes, "card": MODEL_CARD, "max_side": MAX_SIDE}


def _png_base64(mask: np.ndarray) -> str:
    img = Image.frombytes("L", (mask.shape[1], mask.shape[0]), np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/segment")
async def segment(image: UploadFile = File(...), flip: bool = Query(True, description="flip averaging, as in the reported results")) -> dict:
    data = await image.read()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"The photo is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        rgb = load_image(io.BytesIO(data))
    except Exception:  # noqa: BLE001 - any unreadable upload gets the same answer
        raise HTTPException(415, "This file could not be read as a photo (JPEG, PNG or WebP).") from None
    h, w = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:  # the network works at 576x384 internally, so a smaller copy loses nothing it uses
        rgb = np.asarray(Image.fromarray(rgb).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS))
    t0 = time.perf_counter()
    with LOCK:
        SEG.flip_tta = flip
        mask = SEG.predict(rgb)
    ms = (time.perf_counter() - t0) * 1000
    counts = np.bincount(mask.ravel(), minlength=SEG.num_classes)
    found = [{"id": i, "name": SEG.names[i], "percent": round(100 * int(c) / mask.size, 2), "garment": i in GARMENTS}
             for i, c in enumerate(counts) if c and i != 0]
    found.sort(key=lambda x: -x["percent"])
    return {
        "width": int(mask.shape[1]), "height": int(mask.shape[0]),
        "original_width": int(w), "original_height": int(h),
        "mask_png": _png_base64(mask), "classes": found, "ms": round(ms), "flip": flip,
    }
