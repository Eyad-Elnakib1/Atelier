// Colour of each garment, read from the photo pixels inside its mask.
// Colours are compared in OKLab (Ottosson, 2020: https://bottosson.github.io/posts/oklab/),
// where a straight-line distance is close to how different two colours look.

import type { RGB } from "./classes";
import type { Prepared } from "./compose";

export type Lab = [number, number, number];

const LIN = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  LIN[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function toLab(r: number, g: number, b: number): Lab {
  const R = LIN[r | 0], G = LIN[g | 0], B = LIN[b | 0];
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

/** OKLab to sRGB, clipped to the screen's range: fast, for whole images. */
export function labToRgb([L, a, b]: Lab): RGB {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const out = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return out.map((c) => Math.round(gammaOut(Math.min(1, Math.max(0, c))))) as RGB;
}

function gammaOut(c: number): number {
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return v * 255;
}

/** OKLab to sRGB; null when the colour is outside what a screen can show. */
function labToRgbRaw([L, a, b]: Lab): RGB | null {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  if (rgb.some((c) => c < -0.001 || c > 1.001)) return null;
  return rgb.map((c) => Math.round(Math.min(255, Math.max(0, gammaOut(Math.min(1, Math.max(0, c))))))) as RGB;
}

/** A colour from lightness, chroma and hue (OKLCh), with chroma reduced until it fits on screen. */
export function fromLch(L: number, C: number, hDeg: number): RGB {
  const h = (hDeg * Math.PI) / 180;
  for (let c = C; c >= 0; c -= 0.005) {
    const rgb = labToRgbRaw([L, c * Math.cos(h), c * Math.sin(h)]);
    if (rgb) return rgb;
  }
  return labToRgbRaw([L, 0, 0]) ?? [128, 128, 128];
}

export const chroma = (lab: Lab) => Math.hypot(lab[1], lab[2]);
export const hue = (lab: Lab) => ((Math.atan2(lab[2], lab[1]) * 180) / Math.PI + 360) % 360;
export const dist = (p: Lab, q: Lab) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
export const hueGap = (a: number, b: number) => {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
};
export const hex = (c: RGB) => `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;

// Everyday colour names (the site's own list; the hex values are judgement, picked to be typical).
const NAMED: [string, string][] = [
  ["black", "#1c1c1c"], ["charcoal", "#3d3d3f"], ["grey", "#808080"], ["light grey", "#c4c4c4"], ["white", "#f4f4f2"],
  ["cream", "#f0e6cf"], ["beige", "#d7c3a1"], ["stone", "#b3a898"], ["taupe", "#8b7b70"], ["camel", "#b8894f"], ["tan", "#c9a37a"],
  ["light brown", "#a07052"], ["cognac", "#8e4a23"], ["brown", "#6e4630"], ["dark brown", "#3f2a1f"],
  ["navy", "#1f2a46"], ["blue", "#2f5fb3"], ["light blue", "#a3c4e6"], ["denim blue", "#4b6a8e"], ["light denim", "#7f98b8"],
  ["teal", "#1f7a7a"], ["turquoise", "#3fc0be"], ["green", "#3b8a3e"], ["dark green", "#1f4a2e"], ["olive", "#6b6a2e"],
  ["khaki", "#b4a66b"], ["mint", "#a9dcc2"], ["yellow", "#f2d43b"], ["mustard", "#c99a1f"], ["orange", "#e8772e"],
  ["rust", "#a3482a"], ["coral", "#f07b6b"], ["red", "#c1272d"], ["burgundy", "#6c1b2b"], ["pink", "#f2a7ba"],
  ["hot pink", "#e0457b"], ["purple", "#6a3d90"], ["lavender", "#b9a6d7"], ["plum", "#5b2a4b"],
];
const NAMED_LAB = NAMED.map(([name, h]) => ({ name, lab: toLab(parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)) }));

/** Fashion neutrals: they go with almost anything (judgement, common styling convention). */
const NEUTRAL_NAMES = new Set(["black", "charcoal", "grey", "light grey", "white", "cream", "beige", "stone", "taupe", "camel", "tan", "light brown", "cognac", "brown", "dark brown", "navy", "denim blue", "light denim", "khaki"]);
const GREYS = new Set(["black", "charcoal", "grey", "light grey", "white"]);

/**
 * Nearest name, with hue and chroma counted twice as much as lightness (shadows change lightness
 * most; weight 2 is judgement, checked on the sample photos). Near-colourless pixels get a grey name.
 */
export function nameOf(lab: Lab): string {
  let best = NAMED_LAB[0].name, bd = Infinity;
  const grey = chroma(lab) < 0.02;
  for (const n of NAMED_LAB) {
    if (grey && !GREYS.has(n.name)) continue;
    const d = Math.sqrt((lab[0] - n.lab[0]) ** 2 + 4 * ((lab[1] - n.lab[1]) ** 2 + (lab[2] - n.lab[2]) ** 2));
    if (d < bd) { bd = d; best = n.name; }
  }
  return best;
}

export interface Colour {
  rgb: RGB;
  lab: Lab;
  name: string;
  share: number; // part of the garment's pixels in this colour, 0..1
  neutral: boolean;
}

export interface PieceColour {
  main: Colour;
  second: Colour | null; // a second clearly different colour covering a good part of the piece
  patterned: boolean;
}

function makeColour(lab: Lab, share: number): Colour {
  const name = nameOf(lab);
  return { rgb: fromLch(lab[0], chroma(lab), hue(lab)), lab, name, share, neutral: chroma(lab) < 0.04 || NEUTRAL_NAMES.has(name) };
}


function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** k-means in OKLab with k-means++ seeding and a fixed seed, so a photo always gives the same answer. */
function kmeans(pts: Float32Array, n: number, k: number): { centre: Lab; count: number }[] {
  const rand = rng(7);
  const cent: Lab[] = [];
  const first = Math.floor(rand() * n);
  cent.push([pts[first * 3], pts[first * 3 + 1], pts[first * 3 + 2]]);
  const d2 = new Float32Array(n).fill(Infinity);
  while (cent.length < k) {
    const c = cent[cent.length - 1];
    let total = 0;
    for (let i = 0; i < n; i++) {
      const d = (pts[i * 3] - c[0]) ** 2 + (pts[i * 3 + 1] - c[1]) ** 2 + (pts[i * 3 + 2] - c[2]) ** 2;
      if (d < d2[i]) d2[i] = d;
      total += d2[i];
    }
    let r = rand() * total, pick = 0;
    for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { pick = i; break; } }
    cent.push([pts[pick * 3], pts[pick * 3 + 1], pts[pick * 3 + 2]]);
  }
  const assign = new Uint8Array(n);
  for (let it = 0; it < 14; it++) {
    const sum = cent.map(() => [0, 0, 0, 0]);
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let j = 0; j < k; j++) {
        const c = cent[j];
        const d = (pts[i * 3] - c[0]) ** 2 + (pts[i * 3 + 1] - c[1]) ** 2 + (pts[i * 3 + 2] - c[2]) ** 2;
        if (d < bd) { bd = d; best = j; }
      }
      assign[i] = best;
      const s = sum[best];
      s[0] += pts[i * 3]; s[1] += pts[i * 3 + 1]; s[2] += pts[i * 3 + 2]; s[3]++;
    }
    sum.forEach((s, j) => { if (s[3]) cent[j] = [s[0] / s[3], s[1] / s[3], s[2] / s[3]]; });
  }
  const counts = new Array(k).fill(0);
  for (let i = 0; i < n; i++) counts[assign[i]]++;
  return cent.map((centre, j) => ({ centre, count: counts[j] })).filter((c) => c.count > 0);
}

/**
 * Main colour (and a second one, if the piece is two-coloured or patterned) of one class.
 * Pixels on the mask outline are skipped: that is where the model's errors are (report, section 4.2),
 * and where skin or background colour leaks in.
 */
export function pieceColour(p: Prepared, id: number): PieceColour | null {
  const { w, h, labels, photo } = p;
  const inside: number[] = [];
  const r = Math.max(2, Math.round(Math.max(w, h) / 300));
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      if (labels[i] !== id) continue;
      if (labels[i - r] !== id || labels[i + r] !== id || labels[i - r * w] !== id || labels[i + r * w] !== id) continue;
      inside.push(i);
    }
  }
  let idx = inside;
  if (idx.length < 40) { // a thin piece (belt, sunglasses): use every pixel
    idx = [];
    for (let i = 0; i < labels.length; i++) if (labels[i] === id) idx.push(i);
  }
  if (idx.length < 12) return null;
  const step = Math.max(1, Math.floor(idx.length / 5000));
  const n = Math.floor(idx.length / step);
  const pts = new Float32Array(n * 3);
  for (let j = 0; j < n; j++) {
    const o = idx[j * step] * 4;
    const lab = toLab(photo[o], photo[o + 1], photo[o + 2]);
    pts[j * 3] = lab[0]; pts[j * 3 + 1] = lab[1]; pts[j * 3 + 2] = lab[2];
  }
  let groups = kmeans(pts, n, Math.min(4, n));
  // Merge groups that look alike (shading of one fabric); 0.09 in OKLab is judgement.
  let merged = true;
  while (merged && groups.length > 1) {
    merged = false;
    groups.sort((a, b) => b.count - a.count);
    outer: for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        if (dist(groups[a].centre, groups[b].centre) < 0.09) {
          const A = groups[a], B = groups[b], t = A.count + B.count;
          A.centre = A.centre.map((v, i) => (v * A.count + B.centre[i] * B.count) / t) as Lab;
          A.count = t;
          groups.splice(b, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  groups.sort((a, b) => b.count - a.count);
  const main = makeColour(groups[0].centre, groups[0].count / n);
  const g2 = groups[1];
  const second = g2 && g2.count / n >= 0.22 && dist(g2.centre, groups[0].centre) >= 0.15 ? makeColour(g2.centre, g2.count / n) : null;
  const patterned = !!second;
  // A piece with clearly different colours goes with the colours, not with the whites or darks (judgement).
  return { main, second: second && second.name !== main.name ? second : null, patterned };
}

// ---------- colour harmony ----------
// Hue relations on the colour wheel (Itten, "The Art of Color", 1961; the harmonic hue templates
// used by Cohen-Or et al., "Color Harmonization", SIGGRAPH 2006). The degree ranges are judgement.

export type Relation = "neutral" | "accent" | "tonal" | "analogous" | "complementary" | "contrast" | "clash" | "busy";

export interface Verdict {
  relation: Relation;
  title: string;
  text: string;
  good: boolean;
}

// OKLab hue angles are not spaced like the painter's colour wheel the harmony rules are written for
// (on that wheel red is opposite green, and orange opposite blue). Six anchor colours, 60° apart on
// Itten's wheel, map one onto the other; between anchors the mapping is linear.
const ANCHORS = (["#c1272d", "#e8772e", "#f2d43b", "#3b8a3e", "#2f5fb3", "#6a3d90"] as const).map((h) =>
  hue(toLab(parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16))),
);
const WHEEL = [0, 60, 120, 180, 240, 300];

function circular(x: number, from: number[], to: number[]): number {
  const n = from.length;
  for (let i = 0; i < n; i++) {
    const a0 = from[i], a1 = i + 1 < n ? from[i + 1] : from[0] + 360;
    const b0 = to[i], b1 = i + 1 < n ? to[i + 1] : to[0] + 360;
    const v = x < a0 ? x + 360 : x;
    if (v >= a0 && v < a1) return (b0 + ((v - a0) / (a1 - a0)) * (b1 - b0)) % 360;
  }
  return x;
}

/** Position on the painter's colour wheel (red 0°, yellow 120°, blue 240°) of an OKLab hue. */
export const wheel = (okHue: number) => circular(((okHue % 360) + 360) % 360, ANCHORS, WHEEL);
/** OKLab hue of a colour-wheel position (the inverse of wheel). */
export const unwheel = (w: number) => circular(((w % 360) + 360) % 360, WHEEL, ANCHORS);

/** Relation of two hues, both given as colour-wheel positions. */
export function relate(h1: number, h2: number): "analogous" | "complementary" | "contrast" | "clash" {
  const d = hueGap(h1, h2);
  if (d <= 60) return "analogous";
  if (d >= 150) return "complementary";
  if (d >= 100) return "contrast";
  return "clash";
}

export function judge(colours: Colour[]): Verdict {
  const bright = colours.filter((c) => !c.neutral);
  const names = (list: Colour[]) => [...new Set(list.map((c) => c.name))].join(", ");
  if (colours.length === 0) return { relation: "neutral", title: "No clothes found", text: "There is no garment colour to check.", good: false };
  if (bright.length === 0) {
    const Ls = colours.map((c) => c.lab[0]);
    const spread = Math.max(...Ls) - Math.min(...Ls);
    return spread < 0.12
      ? { relation: "tonal", title: "Tonal neutrals", text: `All neutral and close in lightness (${names(colours)}). Calm and easy; one accent colour would add interest.`, good: true }
      : { relation: "neutral", title: "Neutral outfit", text: `All neutral colours (${names(colours)}). They go together; one accent colour would make it stand out.`, good: true };
  }
  if (bright.length === 1 || new Set(bright.map((c) => c.name)).size === 1) {
    return { relation: "accent", title: "Neutral base, one accent", text: `${cap(bright[0].name)} is the only strong colour, set against neutrals. A classic, safe combination.`, good: true };
  }
  const hues = bright.map((c) => wheel(hue(c.lab)));
  const pairs: string[] = [];
  for (let i = 0; i < hues.length; i++) for (let j = i + 1; j < hues.length; j++) pairs.push(relate(hues[i], hues[j]));
  if (pairs.every((p) => p === "analogous")) {
    return { relation: "analogous", title: "Analogous colours", text: `${cap(names(bright))} sit next to each other on the colour wheel, so they blend smoothly.`, good: true };
  }
  if (bright.length > 2) {
    return { relation: "busy", title: "Many strong colours", text: `${bright.length} strong colours (${names(bright)}). It can look busy; swapping one piece for a neutral calms it down.`, good: false };
  }
  const p = pairs[0];
  if (p === "complementary") return { relation: "complementary", title: "Complementary colours", text: `${cap(bright[0].name)} and ${bright[1].name} are opposite on the colour wheel: bold, high contrast. It works best when one of them covers less area.`, good: true };
  if (p === "contrast") return { relation: "contrast", title: "Contrasting colours", text: `${cap(bright[0].name)} and ${bright[1].name} are far apart on the colour wheel: lively, a deliberate contrast.`, good: true };
  return { relation: "clash", title: "Possible clash", text: `${cap(bright[0].name)} and ${bright[1].name} are neither close nor opposite on the colour wheel, which often reads as a clash. A neutral in place of one of them is safer.`, good: false };
}

/** True when two pieces' main colours go together by the rules above. */
export function goesWith(a: Colour, b: Colour): boolean {
  if (a.neutral || b.neutral) return true;
  return relate(wheel(hue(a.lab)), wheel(hue(b.lab))) !== "clash";
}

/** Colour suggestions for one piece: neutrals plus related hues. */
export function partners(c: Colour): { label: string; colours: { name: string; rgb: RGB }[] }[] {
  const neutrals = ["white", "cream", "beige", "grey", "navy", "black"].filter((n) => n !== c.name).map((name) => ({ name, rgb: rgbOfName(name) }));
  if (c.neutral) {
    return [
      { label: "Neutral: goes with almost anything", colours: neutrals.slice(0, 4) },
      { label: "Accent colours to try", colours: [0, 60, 180, 240].map((w) => named(fromLch(0.6, 0.15, unwheel(w)))) },
    ];
  }
  const L = Math.min(0.75, Math.max(0.45, c.lab[0])), C = Math.max(0.08, Math.min(0.14, chroma(c.lab)));
  const w = wheel(hue(c.lab));
  const at = (dw: number) => named(fromLch(L, C, unwheel(w + dw)));
  return [
    { label: "Neutrals", colours: neutrals.slice(0, 4) },
    { label: "Neighbours on the wheel", colours: [at(-40), at(40)] },
    { label: "Opposite on the wheel", colours: [at(180)] },
  ];
}

function named(rgb: RGB) {
  return { name: nameOf(toLab(...rgb)), rgb };
}

function rgbOfName(name: string): RGB {
  const h = NAMED.find((n) => n[0] === name)![1];
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}

export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
