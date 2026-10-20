// Picture tools built on the label map: the fabric studio's patterns and try-on.
// Everything runs in the browser on the photo and mask the page already holds.

import type { RGB } from "./classes";
import { labToRgb, toLab } from "./colour";
import { canvas, type Prepared } from "./compose";

/** The photo as a canvas, at the size the model worked at. */
export function photoCanvas(p: Prepared): HTMLCanvasElement {
  const [c, ctx] = canvas(p.w, p.h);
  ctx.putImageData(new ImageData(new Uint8ClampedArray(p.photo), p.w, p.h), 0, 0);
  return c;
}

export function download(c: HTMLCanvasElement, name: string) {
  const a = document.createElement("a");
  a.href = c.toDataURL("image/png");
  a.download = name;
  a.click();
}

// ---------- fabric patterns ----------

export type Pattern =
  | "solid" | "breton" | "pinstripe" | "gingham" | "windowpane" | "tartan"
  | "houndstooth" | "herringbone" | "chevron" | "argyle" | "polka" | "denim";

export const PATTERNS: { id: Pattern; label: string }[] = [
  { id: "solid", label: "Plain dye" },
  { id: "breton", label: "Breton" },
  { id: "pinstripe", label: "Pinstripe" },
  { id: "gingham", label: "Gingham" },
  { id: "windowpane", label: "Windowpane" },
  { id: "tartan", label: "Tartan" },
  { id: "houndstooth", label: "Houndstooth" },
  { id: "herringbone", label: "Herringbone" },
  { id: "chevron", label: "Chevron" },
  { id: "argyle", label: "Argyle" },
  { id: "polka", label: "Polka dot" },
  { id: "denim", label: "Denim twill" },
];

const rgba = (c: RGB, a = 1) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;

function noise(ctx: CanvasRenderingContext2D, s: number, amount: number) {
  const img = ctx.getImageData(0, 0, s, s);
  let seed = 11;
  for (let i = 0; i < img.data.length; i += 4) {
    seed = (seed * 16807) % 2147483647;
    const n = (seed / 2147483647 - 0.5) * amount;
    img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n;
  }
  ctx.putImageData(img, 0, 0);
}

/** One repeat of a fabric pattern, s pixels square, in a main colour a and a second colour b. */
export function patternTile(kind: Pattern, a: RGB, b: RGB, s: number): HTMLCanvasElement {
  s = Math.max(8, Math.round(s));
  const [c, ctx] = canvas(s, s);
  ctx.fillStyle = rgba(a);
  ctx.fillRect(0, 0, s, s);
  ctx.fillStyle = rgba(b);
  ctx.strokeStyle = rgba(b);
  if (kind === "breton") {
    ctx.fillRect(0, s * 0.62, s, s * 0.38);
  } else if (kind === "pinstripe") {
    ctx.fillRect(s * 0.47, 0, Math.max(1, s * 0.05), s);
  } else if (kind === "gingham") {
    ctx.fillStyle = rgba(b, 0.5);
    ctx.fillRect(0, 0, s / 2, s);
    ctx.fillRect(0, 0, s, s / 2);
  } else if (kind === "windowpane") {
    const t = Math.max(1, s * 0.045);
    ctx.fillRect(0, 0, t, s);
    ctx.fillRect(0, 0, s, t);
  } else if (kind === "tartan") {
    ctx.fillStyle = rgba(b, 0.5);
    ctx.fillRect(s * 0.1, 0, s * 0.3, s);
    ctx.fillRect(0, s * 0.1, s, s * 0.3);
    ctx.fillStyle = rgba(b, 0.85);
    ctx.fillRect(s * 0.68, 0, s * 0.05, s);
    ctx.fillRect(0, s * 0.68, s, s * 0.05);
    ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
    ctx.fillRect(s * 0.86, 0, s * 0.02, s);
    ctx.fillRect(0, s * 0.86, s, s * 0.02);
  } else if (kind === "houndstooth") {
    // 8x8 cells: dark top-left block, light bottom-right block, broken diagonals in the other two.
    const u = s / 8;
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const lx = x % 4, ly = y % 4;
        const dark = x < 4 && y < 4 ? true : x >= 4 && y >= 4 ? false : (lx + 3 - ly) % 4 < 2;
        if (dark) ctx.fillRect(x * u, y * u, u + 0.5, u + 0.5);
      }
    }
  } else if (kind === "herringbone") {
    // Two columns of short slanted strokes leaning opposite ways.
    ctx.lineWidth = Math.max(1, s / 12);
    ctx.lineCap = "butt";
    for (const [x0, dir] of [[0, 1], [s / 2, -1]] as const) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, 0, s / 2, s);
      ctx.clip();
      for (let k = -s; k < 2 * s; k += s / 4) {
        ctx.beginPath();
        ctx.moveTo(x0, k);
        ctx.lineTo(x0 + s / 2, k + (dir * s) / 2);
        ctx.stroke();
      }
      ctx.restore();
    }
    noise(ctx, s, 14);
  } else if (kind === "chevron") {
    ctx.lineWidth = s * 0.16;
    ctx.lineJoin = "miter";
    for (const y of [-s / 2, 0, s / 2, s]) {
      ctx.beginPath();
      ctx.moveTo(0, y + s * 0.25);
      ctx.lineTo(s / 2, y + s * 0.5);
      ctx.lineTo(s, y + s * 0.25);
      ctx.stroke();
    }
  } else if (kind === "argyle") {
    ctx.fillStyle = rgba(b, 0.75);
    ctx.beginPath();
    ctx.moveTo(s / 2, 0);
    ctx.lineTo(s, s / 2);
    ctx.lineTo(s / 2, s);
    ctx.lineTo(0, s / 2);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
    ctx.lineWidth = Math.max(1, s / 40);
    ctx.setLineDash([s / 20, s / 30]);
    ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(s, s);
    ctx.moveTo(s, 0); ctx.lineTo(0, s);
    ctx.stroke();
  } else if (kind === "polka") {
    const r = s * 0.13;
    for (const [x, y] of [[0.25, 0.25], [0.75, 0.75]]) {
      ctx.beginPath();
      ctx.arc(x * s, y * s, r, 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (kind === "denim") {
    ctx.strokeStyle = rgba(b, 0.32);
    ctx.lineWidth = Math.max(1, s / 14);
    for (let k = -s; k < 2 * s; k += s / 5) {
      ctx.beginPath();
      ctx.moveTo(k, 0);
      ctx.lineTo(k + s, s);
      ctx.stroke();
    }
    noise(ctx, s, 26);
  }
  return c;
}

/** A full-size texture made by repeating a tile (a pattern, or a photo of a fabric), turned by angle degrees. */
export function textureFrom(tile: CanvasImageSource, tw: number, th: number, w: number, h: number, angle = 0): Uint8ClampedArray {
  const [t, tctx] = canvas(Math.max(1, Math.round(tw)), Math.max(1, Math.round(th)));
  tctx.imageSmoothingQuality = "high";
  tctx.drawImage(tile, 0, 0, t.width, t.height);
  const [, ctx] = canvas(w, h);
  ctx.fillStyle = ctx.createPattern(t, "repeat")!;
  const d = Math.hypot(w, h);
  ctx.translate(w / 2, h / 2);
  ctx.rotate((angle * Math.PI) / 180);
  ctx.fillRect(-d, -d, 2 * d, 2 * d);
  return ctx.getImageData(0, 0, w, h).data;
}

/** A texture of one flat colour. */
export function solidTexture(c: RGB, w: number, h: number): Uint8ClampedArray {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < d.length; i += 4) { d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255; }
  return d;
}

// ---------- try-on ----------

export type Finish = "matte" | "satin" | "gloss";

/**
 * Put a new fabric on the chosen garments and keep their folds. Works in OKLab: the new pixel takes
 * the fabric's colour, and its lightness is the fabric's lightness plus how much lighter or darker
 * that photo pixel is than its garment's average. Unlike multiplying, this keeps the folds on black
 * and on white clothes alike. The finish scales the shading (matte flatter, gloss with stronger
 * highlights). Approach and factors are judgement, checked on the sample photos.
 */
export function applyFabric(p: Prepared, ids: number[], tex: Uint8ClampedArray, shading = 1, finish: Finish = "satin"): HTMLCanvasElement {
  const { w, h, labels, photo } = p;
  const on = new Uint8Array(256);
  ids.forEach((id) => (on[id] = 1));
  const sum = new Float64Array(256), cnt = new Float64Array(256);
  const L = new Float32Array(w * h);
  for (let i = 0; i < labels.length; i++) {
    const c = labels[i];
    if (!on[c]) continue;
    const o = i * 4;
    L[i] = toLab(photo[o], photo[o + 1], photo[o + 2])[0];
    sum[c] += L[i];
    cnt[c]++;
  }
  const k = shading * (finish === "matte" ? 0.65 : finish === "gloss" ? 1.1 : 1);
  const out = new ImageData(new Uint8ClampedArray(photo), w, h);
  const d = out.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const c = labels[i];
      if (!on[c]) continue;
      const o = i * 4;
      let dev = (L[i] - sum[c] / cnt[c]) * k;
      if (finish === "gloss" && dev > 0) dev *= 1.7;
      const t = toLab(tex[o], tex[o + 1], tex[o + 2]);
      const rgb = labToRgb([Math.min(0.99, Math.max(0.02, t[0] + dev)), t[1], t[2]]);
      // Soften the outline: pixels next to another class take half the new fabric.
      const edge = (x > 0 && labels[i - 1] !== c) || (x < w - 1 && labels[i + 1] !== c) || (y > 0 && labels[i - w] !== c) || (y < h - 1 && labels[i + w] !== c);
      const m = edge ? 0.5 : 1;
      for (let j = 0; j < 3; j++) d[o + j] = photo[o + j] * (1 - m) + rgb[j] * m;
    }
  }
  const [cv, ctx] = canvas(w, h);
  ctx.putImageData(out, 0, 0);
  return cv;
}
