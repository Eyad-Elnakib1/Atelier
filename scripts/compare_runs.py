"""Compare training runs side by side: table, ranking and curves.

Reads ``val_log.csv`` and ``train_log.csv`` of every run in ``<drive-root>/runs``
whose name starts with ``--prefix`` and prints the numbers the bake-off decision is
based on (PLAN.md, section 4.4).

Example (Colab):
    python scripts/compare_runs.py --drive-root /content/drive/MyDrive/clothseg --prefix r1_
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CORE_GARMENTS = ("upper-clothes", "pants", "skirt", "dress")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--prefix", default="r1_", help="Only runs whose folder name starts with this.")
    p.add_argument("--at-epoch", type=int, default=0, help="Compare at this epoch. 0 = the last epoch that every run has reached.")
    p.add_argument("--out", type=Path, default=None, help="PNG file for the curves (default: <drive-root>/reports/compare_<prefix>.png).")
    a = p.parse_args()

    runs = {}
    for d in sorted((a.drive_root / "runs").glob(f"{a.prefix}*")):
        val = read_csv(d / "val_log.csv")
        if val:
            cfg = json.loads((d / "config.json").read_text())
            runs[d.name] = {"val": val, "train": read_csv(d / "train_log.csv"), "model": cfg["model"]}
    if not runs:
        raise SystemExit(f"No runs with a validation log found in {a.drive_root / 'runs'} for prefix '{a.prefix}'.")

    if a.at_epoch:
        # A run that has not reached the requested epoch is listed and left out, not a reason to stop.
        short = {n: len(r["val"]) for n, r in runs.items() if len(r["val"]) < a.at_epoch}
        for n, done in short.items():
            print(f"NOT COMPARED: {n} has only {done} validated epoch(s); it needs {a.at_epoch}. Run its training command again to continue it.")
            del runs[n]
        if not runs:
            raise SystemExit(f"No run has reached epoch {a.at_epoch} yet.")
    common = min(len(r["val"]) for r in runs.values())
    epoch = a.at_epoch or common
    print(f"Comparing {len(runs)} runs at epoch {epoch} (validation set)\n")

    rows = []
    for name, r in runs.items():
        v = r["val"][epoch - 1]
        prev = r["val"][epoch - 2] if epoch >= 2 else None
        speeds = [float(t["img_per_s"]) for t in r["train"] if int(t["epoch"]) <= epoch]
        rows.append({
            "run": name, "garment": float(v["garment_miou"]), "miou": float(v["miou"]), "binary": float(v["clothing_binary_iou"]),
            "core": [float(v[f"iou_{c}"]) for c in CORE_GARMENTS],
            "gain": float(v["garment_miou"]) - float(prev["garment_miou"]) if prev else float("nan"),
            "min_per_epoch": float(v["train_min"]) / epoch, "img_per_s": sum(speeds) / max(1, len(speeds)),
            "epochs_done": len(r["val"]),
        })
    rows.sort(key=lambda x: -x["garment"])

    head = f"{'rank':<5}{'run':<28}{'garment mIoU':>13}{'mIoU':>8}{'cloth IoU':>10}" + "".join(f"{c[:9]:>10}" for c in CORE_GARMENTS) + f"{'last gain':>10}{'min/epoch':>10}{'img/s':>7}"
    print(head)
    print("-" * len(head))
    for i, x in enumerate(rows, 1):
        print(f"{i:<5}{x['run']:<28}{x['garment']:>13.4f}{x['miou']:>8.4f}{x['binary']:>10.4f}" + "".join(f"{c:>10.4f}" for c in x["core"])
              + f"{x['gain']:>+10.4f}{x['min_per_epoch']:>10.1f}{x['img_per_s']:>7.1f}")
    print("\ncolumns: 'cloth IoU' = any garment vs everything else; 'last gain' = garment mIoU change over the last epoch "
          "(a large value means the model is still improving fast); min/epoch = training time only.")

    best = rows[0]
    print("\nReading the table (rules fixed in PLAN.md):")
    print(f"  - Primary ranking is garment mIoU. Leader: {best['run']} ({best['garment']:.4f}).")
    for x in rows[1:]:
        gap = best["garment"] - x["garment"]
        note = "within 1 point of the leader: a tie, the faster model wins" if gap < 0.01 else f"{100 * gap:.1f} points behind"
        print(f"  - {x['run']}: {note}.")
    weak = [(x["run"], c, v) for x in rows for c, v in zip(CORE_GARMENTS, x["core"]) if v < best["core"][CORE_GARMENTS.index(c)] - 0.05]
    for run, c, v in weak:
        print(f"  - CHECK: {run} is more than 5 points behind the leader on '{c}' ({v:.4f}).")

    out = a.out or a.drive_root / "reports" / f"compare_{a.prefix.strip('_') or 'runs'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, r in runs.items():
        ep = [int(v["epoch"]) for v in r["val"]]
        axes[0].plot(ep, [float(v["garment_miou"]) for v in r["val"]], marker="o", label=name)
        axes[1].plot(ep, [float(v["val_loss"]) for v in r["val"]], marker="o", label=name)
        steps = [int(t["step"]) for t in r["train"]]
        loss = [float(t["loss"]) for t in r["train"]]
        k = max(1, len(loss) // 200)  # smooth: average over windows so the curves stay readable
        axes[2].plot([steps[i] for i in range(0, len(loss) - k + 1, k)], [sum(loss[i:i + k]) / k for i in range(0, len(loss) - k + 1, k)], label=name)
    for ax, title, xl in zip(axes, ("Validation garment mIoU", "Validation loss", "Training loss"), ("epoch", "epoch", "step")):
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\ncurves saved: {out}")


if __name__ == "__main__":
    main()
