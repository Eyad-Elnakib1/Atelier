"""Charts for the comparisons in report_latex/report.tex, drawn at their printed size.

Every number comes from the report's own tables (and, for the error breakdown, from the
executed notebook colab_why_ours_wins). Output: assets/charts/*.png

Usage:
    py -3 tools/make_report_charts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

COLUMN, FULL = 3.35, 6.9          # inches: one column and the full text width of the paper
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#8c8b86"
# Validated categorical order (dataviz reference palette, light): fixed per model, never cycled.
MODEL_COLOR = {
    "Ours, B3 30 ep": "#2a78d6",
    "Ours, B3 12 ep": "#eb6834",
    "FASHN": "#1baf7a",
    "YOLO26x-sem": "#eda100",
    "SCHP-ATR": "#e87ba4",
}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Model sizes (Table 2 and the models trained here) -----------------------------------------
sizes = [("YOLO26x-sem", 40.2, "trained"), ("SegFormer-B3 (ours)", 47.2, "ours"), ("FASHN (SegFormer-B4)", 64.1, "compared"),
         ("SCHP (ResNet-101)", 66.8, "compared"), ("Two-encoder (ours, rejected)", 83, "trained"),
         ("EoMT ViT-L", 300, "large"), ("Sapiens 0.3B", 336, "large"), ("SAM 3", 848, "large"), ("Sapiens 1B", 1200, "large")]
fig, ax = plt.subplots(figsize=(COLUMN, 2.25))
y = np.arange(len(sizes))
for i, (name, m, kind) in enumerate(sizes):
    ax.barh(i, m, height=0.62, color=MODEL_COLOR["Ours, B3 30 ep"] if kind == "ours" else (NEUTRAL if kind != "large" else "#c9c8c3"))
    ax.text(m * 1.08, i, f"{m:g} M", va="center", fontsize=6.5, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.set_yticks(y, [s[0] for s in sizes])
ax.set_xscale("log")
ax.set_xlim(20, 4000)
ax.axvline(90, color=INK2, lw=0.8, ls="--")
ax.set_ylim(len(sizes) - 0.5, -1.2)
ax.text(100, -0.85, "too large for one 22 GB L4 (right of line)", fontsize=6.3, color=INK2, va="center")
ax.set_xlabel("parameters (millions, log scale)")
ax.grid(axis="y", visible=False)
save(fig, "model_sizes.png")

# 2. Bake-off round 2: why B3 (Table 3) -------------------------------------------------------
finalists = {"SegFormer-B3": ("#2a78d6", [0.7303, 0.735, 0.730]), "SegFormer-B4": ("#6250d6", [0.7231, 0.718, 0.701]),
             "UPerNet + ConvNeXt-T": ("#008300", [0.7182, 0.710, 0.702])}
metrics = ["garment mIoU", "skirt IoU", "dress IoU"]
fig, ax = plt.subplots(figsize=(COLUMN, 1.55))
for j, (name, (col, vals)) in enumerate(finalists.items()):
    ax.scatter(vals, np.arange(len(metrics)) + (j - 1) * 0.18, s=26, color=col, label=name, zorder=3, edgecolor="white", linewidth=0.8)
for i in range(len(metrics)):
    vals = [v[1][i] for v in finalists.values()]
    ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=1.5, zorder=1)
ax.set_yticks(range(len(metrics)), metrics)
ax.invert_yaxis()
ax.set_xlim(0.695, 0.742)
ax.set_xlabel("ATR validation, epoch 12 (higher is better)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.9)
save(fig, "bakeoff_round2.png")

# 3. Per-class IoU of the final model on the test split (Section 7) ----------------------------
per_class = [("legs", .889, 0), ("face", .858, 0), ("pants", .848, 1), ("arms", .830, 0), ("upper-clothes", .828, 1),
             ("hair", .827, 0), ("bag", .818, 1), ("hat", .789, 1), ("shoes", .770, 1), ("skirt", .746, 1),
             ("dress", .724, 1), ("scarf", .704, 1), ("sunglasses", .688, 1), ("belt", .524, 1)]
fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
for i, (name, v, garment) in enumerate(per_class):
    ax.barh(i, v, height=0.66, color=MODEL_COLOR["Ours, B3 30 ep"] if garment else "#c9c8c3")
    ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK)
ax.set_yticks(range(len(per_class)), [p[0] for p in per_class])
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.set_xlabel("IoU, test split, flip averaging")
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR["Ours, B3 30 ep"]), plt.Rectangle((0, 0), 1, 1, color="#c9c8c3")],
          labels=["garment class", "body class"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
save(fig, "per_class_test.png")

# 4. CCP comparison per garment (Table 6) ------------------------------------------------------
classes = ["upper-\nclothes", "pants", "skirt", "dress", "hat", "sun-\nglasses", "belt", "shoes", "bag", "scarf"]
ccp = {
    "Ours, B3 30 ep": [.835, .871, .401, .475, .760, .722, .497, .821, .786, .592],
    "Ours, B3 12 ep": [.829, .855, .378, .451, .753, .714, .503, .817, .769, .598],
    "FASHN": [.833, .845, .398, .500, .713, .623, .466, np.nan, .781, .605],
    "YOLO26x-sem": [.817, .841, .375, .451, .712, .686, .498, .810, .747, .510],
    "SCHP-ATR": [.779, .747, .339, .356, .652, .565, .388, .769, .719, .525],
}
fig, ax = plt.subplots(figsize=(FULL, 1.9))
x = np.arange(len(classes))
w = 0.16
for j, (name, vals) in enumerate(ccp.items()):
    ax.bar(x + (j - 2) * w, vals, width=w * 0.9, color=MODEL_COLOR[name], label=name)
ax.text(x[7] + 0 * w, 0.03, "no class", rotation=90, fontsize=5.5, color=INK2, ha="center", va="bottom")
ax.set_xticks(x, classes)
ax.set_ylim(0, 1)
ax.set_ylabel("IoU on CCP (986 photos)")
ax.grid(axis="x", visible=False)
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
save(fig, "ccp_per_class.png")

# 5. Error breakdown on CCP (executed notebook colab_why_ours_wins) -----------------------------
models = list(MODEL_COLOR)
outline = [73.444, 72.128, 69.719, 70.014, 67.847]
inside = [87.076, 86.178, 85.424, 82.724, 82.584]
order = ["Ours, B3 30 ep", "Ours, B3 12 ep", "YOLO26x-sem", "SCHP-ATR", "FASHN"]   # same order as the notebook columns
size_bins = {"small (under 1%)": [.624, .618, .606, .517, .365], "medium (1 to 5%)": [.795, .789, .768, .717, .608], "large (over 5%)": [.818, .805, .795, .763, .813]}
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 1.95), gridspec_kw={"width_ratios": [1, 1.25]})
yy = np.arange(len(order))
for i, name in enumerate(order):
    a.plot([outline[i], inside[i]], [i, i], color=GRID, lw=1.5, zorder=1)
    a.scatter(outline[i], i, s=24, marker="o", color=MODEL_COLOR[name], zorder=3, edgecolor="white", linewidth=0.8)
    a.scatter(inside[i], i, s=26, marker="s", color=MODEL_COLOR[name], zorder=3, edgecolor="white", linewidth=0.8)
a.set_yticks(yy, order)
a.invert_yaxis()
a.set_xlim(65, 90)
a.set_xlabel("garment pixel accuracy (%)")
a.set_title("(a) along outlines (circle) and inside regions (square)", loc="left")
a.grid(axis="y", visible=False)
xb = np.arange(len(size_bins))
for j, name in enumerate(order):
    b.bar(xb + (j - 2) * w, [v[j] for v in size_bins.values()], width=w * 0.9, color=MODEL_COLOR[name], label=name)
b.set_xticks(xb, [f"{k}\nof the photo" for k in size_bins])
b.set_ylim(0, 1)
b.set_ylabel("garment IoU")
b.set_title("(b) by garment size", loc="left")
b.grid(axis="x", visible=False)
b.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
save(fig, "ccp_error_breakdown.png")

# 6. ATR against CCP for ours and SCHP (Table 7) ------------------------------------------------
fig, ax = plt.subplots(figsize=(COLUMN, 1.75))
for name, key, vals in (("Ours (B3, 30 ep)", "Ours, B3 30 ep", (0.786, 0.726)), ("SCHP-ATR", "SCHP-ATR", (0.823, 0.634))):
    ax.plot([0, 1], vals, color=MODEL_COLOR[key], lw=2, marker="o", ms=5, markeredgecolor="white")
    ax.text(-0.06, vals[0], f"{vals[0]:.3f}", ha="right", va="center", fontsize=6.8, color=INK)
    ax.text(1.06, vals[1], f"{vals[1]:.3f}  {name}", ha="left", va="center", fontsize=6.8, color=INK)
ax.set_xticks([0, 1], ["ATR test\n(each on its own split)", "CCP\n(same 986 photos)"])
ax.set_xlim(-0.35, 1.75)
ax.set_ylim(0.6, 0.85)
ax.set_ylabel("mIoU, all classes")
ax.grid(axis="x", visible=False)
save(fig, "atr_vs_ccp_schp.png")

# 7. Changes tried after selection (Table 10) ---------------------------------------------------
changes = [("Double learning rate", 0.46), ("Fine-tune at 768x512", 0.26), ("Two-encoder B3 + ConvNeXt-T", 0.12),
           ("Weight-averaged fine-tune", 0.01), ("Ensemble B3 + UPerNet", -0.15), ("SegFormer-B4", -0.72),
           ("Edge branch", -0.90), ("Half learning rate", -1.23), ("ADE20K start", -1.57)]
fig, ax = plt.subplots(figsize=(COLUMN, 2.15))
for i, (name, d) in enumerate(changes):
    ax.barh(i, d, height=0.62, color=NEUTRAL)
    ax.text(d + (0.05 if d >= 0 else -0.05), i, f"{d:+.2f}", va="center", ha="left" if d >= 0 else "right", fontsize=6.5, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.set_yticks(range(len(changes)), [c[0] for c in changes])
ax.invert_yaxis()
ax.axvline(0, color=INK2, lw=0.8)
ax.axvline(0.5, color=MODEL_COLOR["Ours, B3 30 ep"], lw=1, ls="--")
ax.text(0.53, len(changes) - 0.6, "required\n+0.50", fontsize=6.3, color=MODEL_COLOR["Ours, B3 30 ep"], va="bottom")
ax.set_xlim(-2.0, 1.0)
ax.set_xlabel("change in validation garment mIoU (points)")
ax.grid(axis="y", visible=False)
save(fig, "changes_delta.png")
