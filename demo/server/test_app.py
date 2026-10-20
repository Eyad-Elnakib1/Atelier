"""Tests for the demo API. Needs best.pt in the project root (or MODEL_PATH) and a few photos.

Run from the project root:
    py -3 -m pytest demo/server/test_app.py -q          (if pytest is installed)
    py -3 demo/server/test_app.py <folder with .jpg photos>
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from demo.server.app import MAX_SIDE, SEG, app  # noqa: E402

client = TestClient(app)


def decode(mask_b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(mask_b64))))


def check_photo(path: Path) -> None:
    data = path.read_bytes()
    t0 = time.perf_counter()
    r = client.post("/api/segment", files={"image": (path.name, data, "image/jpeg")})
    wall = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, r.text
    j = r.json()
    mask = decode(j["mask_png"])
    assert mask.shape == (j["height"], j["width"]), (mask.shape, j["height"], j["width"])
    assert max(j["width"], j["height"]) <= MAX_SIDE
    # Same answer as calling the pipeline directly on the same reduced photo.
    from clothseg.inference import load_image
    rgb = load_image(path)
    h, w = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        rgb = np.asarray(Image.fromarray(rgb).resize((round(w * scale), round(h * scale)), Image.LANCZOS))
    SEG.flip_tta = True
    direct = SEG.predict(rgb)
    same = float((direct == mask).mean())
    assert same > 0.999, same
    names = ", ".join(f"{c['name']} {c['percent']}%" for c in j["classes"][:6])
    print(f"  {path.name}: {j['width']}x{j['height']} | model {j['ms']} ms, request {wall:.0f} ms | identical to direct call: {same:.4f} | {names}")


def main() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok", r.text
    print("health:", r.json())
    info = client.get("/api/info").json()
    assert len(info["classes"]) == 15 and info["classes"][0]["name"] == "background"
    print("info: 15 classes, garments:", [c["name"] for c in info["classes"] if c["garment"]])

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    photos = sorted(folder.glob("*.jpg"))[:3] if folder else []
    for p in photos:
        check_photo(p)

    # Error cases
    assert client.post("/api/segment", files={"image": ("e.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/api/segment", files={"image": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 415
    assert client.post("/api/segment", files={"image": ("big.jpg", b"0" * (10 * 1024 * 1024 + 1), "image/jpeg")}).status_code == 413
    buf = io.BytesIO()
    Image.new("RGBA", (300, 2000), (200, 100, 50, 128)).save(buf, "PNG")   # transparent, tall, larger than MAX_SIDE
    r = client.post("/api/segment?flip=false", files={"image": ("t.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["height"] == MAX_SIDE and r.json()["flip"] is False, r.text
    print("errors: empty 400, not an image 415, too large 413; tall transparent PNG reduced to", r.json()["width"], "x", r.json()["height"])
    print("ALL PASSED")


if __name__ == "__main__":
    main()
