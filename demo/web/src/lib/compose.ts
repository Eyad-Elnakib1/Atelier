// Everything visual is computed in the browser from two things: the photo and the
// label mask (one class id per pixel). Switching views is therefore instant.

import { BODY_IDS, CLASSES, GARMENT_IDS, type RGB } from "./classes";

export interface Prepared {
  w: number;
  h: number;
  photo: Uint8ClampedArray; // RGBA, w * h * 4
  labels: Uint8Array; // class id per pixel
}

export type View = "photo" | "map" | "clothes" | "person" | "overlay";

const PAPER: RGB = [239, 232, 220];
const INK: RGB = [20, 17, 15];

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("The photo could not be displayed by this browser."));
    img.src = src;
  });
}

export function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("This browser cannot draw on a canvas.");
  return [c, ctx];
}

/** Photo and mask brought to the same size (the size the model worked at). */
export async function prepare(photoSrc: string, maskPngBase64: string, w: number, h: number): Promise<Prepared> {
  const [photoImg, maskImg] = await Promise.all([loadImage(photoSrc), loadImage(`data:image/png;base64,${maskPngBase64}`)]);
  const [, pctx] = canvas(w, h);
  pctx.imageSmoothingQuality = "high";
  pctx.drawImage(photoImg, 0, 0, w, h);
  const photo = pctx.getImageData(0, 0, w, h).data;
  const [, mctx] = canvas(w, h);
  mctx.imageSmoothingEnabled = false;
  mctx.drawImage(maskImg, 0, 0, w, h);
  const m = mctx.getImageData(0, 0, w, h).data;
  const labels = new Uint8Array(w * h);
  for (let i = 0; i < labels.length; i++) labels[i] = m[i * 4];
  return { w, h, photo, labels };
}

const GARMENT = new Uint8Array(256);
GARMENT_IDS.forEach((id) => (GARMENT[id] = 1));
const BODY = new Uint8Array(256);
BODY_IDS.forEach((id) => (BODY[id] = 1));

/** One view as RGBA pixels. With `focus`, every other class is dimmed. */
export function render(p: Prepared, view: View, focus: number | null = null): ImageData {
  const out = new ImageData(p.w, p.h);
  const d = out.data;
  const { photo, labels } = p;
  for (let i = 0; i < labels.length; i++) {
    const k = labels[i];
    const o = i * 4;
    let r = photo[o], g = photo[o + 1], b = photo[o + 2];
    const col = CLASSES[k]?.color ?? INK;
    if (view === "map") {
      [r, g, b] = k === 0 ? PAPER : col;
    } else if (view === "clothes") {
      if (!GARMENT[k]) [r, g, b] = INK;
    } else if (view === "person") {
      if (k === 0) [r, g, b] = INK;
    } else if (view === "overlay") {
      if (k === 0) {
        r *= 0.38; g *= 0.38; b *= 0.38;
      } else {
        const a = GARMENT[k] ? 0.62 : 0.35;
        r = r * (1 - a) + col[0] * a;
        g = g * (1 - a) + col[1] * a;
        b = b * (1 - a) + col[2] * a;
      }
    }
    if (focus !== null && k !== focus) {
      const t = 0.78;
      r = r * (1 - t) + INK[0] * t;
      g = g * (1 - t) + INK[1] * t;
      b = b * (1 - t) + INK[2] * t;
    }
    d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
  }
  return out;
}

/** Photo pixels of the given classes on a transparent background, cropped to their box. */
export function cutout(p: Prepared, ids: number[], pad = 6): { url: string; w: number; h: number } | null {
  const c = cutoutCanvas(p, ids, pad);
  return c ? { url: c.toDataURL("image/png"), w: c.width, h: c.height } : null;
}

/** The same cut-out as a canvas, for further drawing. */
export function cutoutCanvas(p: Prepared, ids: number[], pad = 6): HTMLCanvasElement | null {
  const want = new Uint8Array(256);
  ids.forEach((id) => (want[id] = 1));
  let x0 = p.w, y0 = p.h, x1 = -1, y1 = -1;
  for (let y = 0; y < p.h; y++) {
    for (let x = 0; x < p.w; x++) {
      if (want[p.labels[y * p.w + x]]) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
  x1 = Math.min(p.w - 1, x1 + pad); y1 = Math.min(p.h - 1, y1 + pad);
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  const [c, ctx] = canvas(w, h);
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const si = (y0 + y) * p.w + (x0 + x);
      const di = (y * w + x) * 4;
      if (want[p.labels[si]]) {
        img.data[di] = p.photo[si * 4];
        img.data[di + 1] = p.photo[si * 4 + 1];
        img.data[di + 2] = p.photo[si * 4 + 2];
        img.data[di + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Full-size transparent layer holding only the given classes (for the exploded view). */
export function layer(p: Prepared, ids: number[], tint: RGB | null = null): string {
  const want = new Uint8Array(256);
  ids.forEach((id) => (want[id] = 1));
  const [c, ctx] = canvas(p.w, p.h);
  const img = ctx.createImageData(p.w, p.h);
  for (let i = 0; i < p.labels.length; i++) {
    if (!want[p.labels[i]]) continue;
    const o = i * 4;
    if (tint) {
      img.data[o] = p.photo[o] * 0.55 + tint[0] * 0.45;
      img.data[o + 1] = p.photo[o + 1] * 0.55 + tint[1] * 0.45;
      img.data[o + 2] = p.photo[o + 2] * 0.55 + tint[2] * 0.45;
    } else {
      img.data[o] = p.photo[o];
      img.data[o + 1] = p.photo[o + 1];
      img.data[o + 2] = p.photo[o + 2];
    }
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/png");
}

export function present(p: Prepared): Set<number> {
  const s = new Set<number>();
  for (let i = 0; i < p.labels.length; i++) s.add(p.labels[i]);
  return s;
}
