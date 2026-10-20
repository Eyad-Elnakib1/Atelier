"""Why one model scores lower than another: where its errors are.

A ``Diagnosis`` collects, over many photos, four kinds of evidence for one model:

  confusion  what each garment's pixels were predicted as (the right class, background, or another class)
  outline    accuracy in a thin band along the outline of every labelled region, and inside the regions
  size       IoU of garments grouped by how much of the photo they cover
  presence   garments missed altogether, and garments invented where none is labelled

Comparing two diagnoses (``explain``) says which of these accounts for a score gap, for
example "dress pixels predicted as upper-clothes: 12% against 7%" (the model cannot tell a
dress from a top) or "accuracy along outlines: 71% against 78%" (coarser output).
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE_BINS = (("small (<1% of photo)", 0.0, 0.01), ("medium (1-5%)", 0.01, 0.05), ("large (>5%)", 0.05, 1.01))
PRESENT_MIN = 0.002   # a garment counts as present in a photo from 0.2% of its labelled pixels
MISSED_RECALL = 0.2   # present, but less than 20% of it found: missed


class Diagnosis:
    def __init__(self, names: list[str] | tuple[str, ...], garments: list[str] | tuple[str, ...]) -> None:
        self.names = list(names)
        self.garments = [self.names.index(g) for g in garments]
        k = self.k = len(self.names)
        self.cm = np.zeros((k, k), np.int64)
        self.cm_outline = np.zeros((k, k), np.int64)
        self.cm_inside = np.zeros((k, k), np.int64)
        self.size = {b[0]: np.zeros(2, np.int64) for b in SIZE_BINS}   # intersection, union
        self.present = np.zeros(k, np.int64)
        self.missed = np.zeros(k, np.int64)
        self.absent = np.zeros(k, np.int64)
        self.invented = np.zeros(k, np.int64)
        self.per_image: dict[str, dict[str, float]] = {}   # photo -> garment IoU per class present

    def _bincount(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return np.bincount(gt.astype(np.int64) * self.k + pred, minlength=self.k * self.k).reshape(self.k, self.k)

    def update(self, gt: np.ndarray, pred: np.ndarray, key: str | int | None = None) -> None:
        """gt: labels with 255 = ignored; pred: predicted labels, same size."""
        valid = gt != 255
        self.cm += self._bincount(gt[valid], pred[valid])
        # Outline band: pixels whose neighbourhood holds more than one label, about 0.4% of the long side wide.
        r = max(2, round(0.004 * max(gt.shape)))
        kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        band = cv2.dilate(gt, kernel) != cv2.erode(gt, kernel)
        self.cm_outline += self._bincount(gt[valid & band], pred[valid & band])
        self.cm_inside += self._bincount(gt[valid & ~band], pred[valid & ~band])

        n_valid = max(1, int(valid.sum()))
        ious: dict[str, float] = {}
        for c in self.garments:
            g, p = (gt == c) & valid, (pred == c) & valid
            n_g, n_p = int(g.sum()), int(p.sum())
            inter, union = int((g & p).sum()), int((g | p).sum())
            if n_g:
                frac = n_g / n_valid
                for name, lo, hi in SIZE_BINS:
                    if lo <= frac < hi:
                        self.size[name] += (inter, union)
                if frac >= PRESENT_MIN:
                    self.present[c] += 1
                    self.missed[c] += inter / n_g < MISSED_RECALL
            else:
                self.absent[c] += 1
                self.invented[c] += n_p / n_valid >= PRESENT_MIN
            if union:
                ious[self.names[c]] = inter / union
        if key is not None:
            self.per_image[str(key)] = ious

    def image_score(self, key: str | int, exclude: tuple[str, ...] = ()) -> float:
        """Garment mIoU of one photo, over the garments labelled or predicted in it, leaving out ``exclude``."""
        vals = [v for n, v in self.per_image.get(str(key), {}).items() if n not in exclude]
        return float(np.mean(vals)) if vals else float("nan")

    def worst_against(self, ref: "Diagnosis", n: int = 6, exclude: tuple[str, ...] = ()) -> list[tuple[float, str]]:
        """The n photos where this model falls furthest behind ``ref``: (gap, photo key), largest gap first."""
        gaps = [(ref.image_score(k, exclude) - self.image_score(k, exclude), k) for k in self.per_image if k in ref.per_image]
        return sorted((g for g in gaps if not np.isnan(g[0])), reverse=True)[:n]

    # ------------------------------------------------------------------ summaries
    @staticmethod
    def _iou(cm: np.ndarray) -> np.ndarray:
        inter = np.diag(cm).astype(float)
        union = cm.sum(0) + cm.sum(1) - np.diag(cm)
        return np.where(union > 0, inter / np.maximum(union, 1), np.nan)

    def garment_miou(self, cm: np.ndarray | None = None) -> float:
        iou = self._iou(self.cm if cm is None else cm)
        return float(np.nanmean(iou[self.garments]))

    def where_pixels_go(self, c: int) -> dict[str, float]:
        """Share of class c's labelled pixels predicted as each class."""
        row = self.cm[c].astype(float)
        return {self.names[j]: row[j] / row.sum() for j in range(self.k)} if row.sum() else {}

    def confusion_rows(self) -> list[dict]:
        rows = []
        for c in self.garments:
            share = self.where_pixels_go(c)
            if not share:
                continue
            name = self.names[c]
            others = sorted(((v, n) for n, v in share.items() if n not in (name, self.names[0])), reverse=True)[:2]
            rows.append({"garment": name, "correct": share[name], "as background": share[self.names[0]],
                         "most confused with": ", ".join(f"{n} {100 * v:.1f}%" for v, n in others)})
        return rows

    def outline_rows(self) -> dict:
        def acc(cm):
            g = self.garments
            return float(np.diag(cm)[g].sum() / max(1, cm[g].sum()))
        return {"garment pixel accuracy along outlines": acc(self.cm_outline),
                "garment pixel accuracy inside regions": acc(self.cm_inside),
                "share of garment pixels on outlines": float(self.cm_outline[self.garments].sum() / max(1, self.cm[self.garments].sum()))}

    def size_rows(self) -> dict:
        return {name: (iu[0] / iu[1] if iu[1] else float("nan")) for name, iu in self.size.items()}

    def presence_rows(self) -> list[dict]:
        return [{"garment": self.names[c], "photos with it": int(self.present[c]),
                 "missed (%)": 100 * self.missed[c] / max(1, self.present[c]),
                 "invented (% of photos without it)": 100 * self.invented[c] / max(1, self.absent[c])}
                for c in self.garments]


def explain(a: Diagnosis, b: Diagnosis, name_a: str, name_b: str, top: int = 4) -> list[str]:
    """Plain sentences on why model a scores differently from model b (same photos, same classes)."""
    lines = []
    ga, gb = a.garment_miou(), b.garment_miou()
    lines.append(f"garment mIoU: {name_a} {ga:.3f}, {name_b} {gb:.3f} ({100 * (ga - gb):+.1f} points)")
    ia, ib = Diagnosis._iou(a.cm), Diagnosis._iou(b.cm)
    gaps = sorted((ia[c] - ib[c], c) for c in a.garments if not (np.isnan(ia[c]) or np.isnan(ib[c])))
    for d, c in gaps[:top]:
        if d >= 0:
            break
        name = a.names[c]
        sa, sb = a.where_pixels_go(c), b.where_pixels_go(c)
        moves = sorted(((sa[n] - sb[n], n) for n in sa if n != name), reverse=True)[:2]
        moved = "; ".join(f"labelled {name} predicted as {n}: {100 * sa[n]:.1f}% against {100 * sb[n]:.1f}%" for dv, n in moves if dv > 0.002)
        extra = []
        pa = 100 * a.invented[c] / max(1, a.absent[c]); pb = 100 * b.invented[c] / max(1, b.absent[c])
        if pa - pb >= 1:
            extra.append(f"{name} invented in {pa:.1f}% of photos without it, against {pb:.1f}%")
        lines.append(f"{name}: IoU {ia[c]:.3f} against {ib[c]:.3f} ({100 * d:+.1f}). " + (moved or "no single class takes the pixels")
                     + ("; " + "; ".join(extra) if extra else ""))
    oa, ob = a.outline_rows(), b.outline_rows()
    for key in ("garment pixel accuracy along outlines", "garment pixel accuracy inside regions"):
        lines.append(f"{key}: {100 * oa[key]:.1f}% against {100 * ob[key]:.1f}% ({100 * (oa[key] - ob[key]):+.1f})")
    sa, sb = a.size_rows(), b.size_rows()
    lines.append("garment IoU by size: " + "; ".join(f"{k} {sa[k]:.3f} against {sb[k]:.3f}" for k in sa))
    return lines
