"""Build notebooks/colab_eval.ipynb, bump the code version and rebuild clothseg_code.zip.

The setup cells (mount, get code, install, prepare data) are copied from
notebooks/colab_train.ipynb so both notebooks behave the same.

Run from the project root:  py -3 tools/build_eval_notebook.py
"""
import json
import zipfile
from pathlib import Path

import nbformat as nbf

VERSION = "19"
Path("VERSION").write_text(VERSION + "\n")

train_path = Path("notebooks/colab_train.ipynb")
train = json.loads(train_path.read_text(encoding="utf-8"))
for cell in train["cells"]:  # the training notebook must ask for the same code version
    src = "".join(cell["source"])
    if "CODE_VERSION = " in src:
        import re
        cell["source"] = re.sub(r'CODE_VERSION = "\d+"', f'CODE_VERSION = "{VERSION}"', src).splitlines(keepends=True)
train_path.write_text(json.dumps(train, indent=1), encoding="utf-8")


def train_cell(marker: str) -> str:
    """Source of the training-notebook code cell that contains ``marker``."""
    hits = ["".join(c["source"]) for c in train["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Clothes segmentation: evaluate a model and segment your own photos

Run this after a training run has produced a `best.pt`. It does not train anything.

1. `Runtime > Change runtime type`: choose **L4 GPU**. The training cells (7 and 8) do not fit a T4; cells 5 and 6 alone also work on a T4.
2. Set `RUN_NAME` in cell 1 to the run you want to look at.
3. Run the cells from top to bottom. Cell 3 asks for the new `clothseg_code.zip` once."""),
    md("## 1. Settings"),
    code(f'''REPO_URL = ""  # leave empty to use the uploaded zip

DRIVE_ROOT = "/content/drive/MyDrive/clothseg"
CODE_DIR = "/content/clothes-segmentation"
CODE_VERSION = "{VERSION}"
HEIGHT, WIDTH = 576, 384
SESSION_BUDGET_MIN = 600

RUN_NAME = "r1_segformer_b3"      # folder name inside <DRIVE_ROOT>/runs
CHECKPOINT = f"{{DRIVE_ROOT}}/runs/{{RUN_NAME}}/checkpoints/best.pt"'''),
    md("## 2. Mount Drive and show the environment"),
    code(train_cell("drive.mount")),
    md("## 3. Get the code"),
    code(train_cell("def code_version")),
    md("## 4. Install requirements and prepare the dataset"),
    code(train_cell("pip install") + '\n!python scripts/prepare_data.py --drive-root "{DRIVE_ROOT}" --height {HEIGHT} --width {WIDTH}'),
    md("""## 5. Evaluate on the validation split

Metrics at the original image resolution, a per-class table, the confusion matrix, and the best and worst predictions."""),
    code('''assert os.path.exists(CHECKPOINT), f"No checkpoint at {CHECKPOINT}. Check RUN_NAME, or wait until the run has finished its first epoch."
OUT = f"{DRIVE_ROOT}/reports/eval_{RUN_NAME}_val"
!python scripts/evaluate.py --checkpoint "{CHECKPOINT}" --split val --out-dir "{OUT}"

from IPython.display import Image, display
for name in ("confusion_matrix.png", "worst_examples.png", "best_examples.png"):
    display(Image(f"{OUT}/{name}"))

# The same evaluation with flip averaging: predict on the photo and on its mirror image, average the two.
import json
!python scripts/evaluate.py --checkpoint "{CHECKPOINT}" --split val --out-dir "{OUT}_flip" --flip-tta > /content/eval_flip.log 2>&1
plain, flip = (json.load(open(f"{d}/metrics.json")) for d in (OUT, f"{OUT}_flip"))
print(f"\\n{'metric':<24}{'plain':>9}{'with flip':>11}{'change':>9}")
for k in ("miou", "garment_miou", "core_garment_miou", "clothing_binary_iou", "boundary_miou", "pixel_acc", "ms_per_image"):
    print(f"{k:<24}{plain[k]:>9.4f}{flip[k]:>11.4f}{flip[k] - plain[k]:>+9.4f}")'''),
    md("""## 6. Segment your own photos

Upload one or more photos of a person. For each photo you get a label mask, a colour overlay and one transparent cut-out per garment."""),
    code('''from google.colab import files
import glob

PHOTOS, RESULTS = "/content/my_photos", f"{DRIVE_ROOT}/reports/segment_{RUN_NAME}"
shutil.rmtree(PHOTOS, ignore_errors=True)
os.makedirs(PHOTOS)
os.chdir(PHOTOS)
uploaded = files.upload()
os.chdir(CODE_DIR)
assert uploaded, "No photo was uploaded."

!python scripts/segment.py --checkpoint "{CHECKPOINT}" --input "{PHOTOS}" --out-dir "{RESULTS}"

for path in sorted(glob.glob(f"{RESULTS}/*_overlay.jpg")):
    print(os.path.basename(path))
    display(Image(path, width=420))'''),
    md("""## 7. Architecture experiment: SegFormer-B3 with an edge branch

Trains SegFormer-B3 with the added edge branch for 12 epochs, with the same data, seed and schedule as the baseline run `r1_segformer_b3`, then prints both side by side. About 2.5 hours. If the session ends, run cells 1 to 4 and this cell again."""),
    code('''EDGE_RUN = "abl_segformer_b3_edge"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 4 and this cell again.")

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/train.py --drive-root "{DRIVE_ROOT}" --run-name {EDGE_RUN} --model segformer_b3_edge --height {HEIGHT} --width {WIDTH} --epochs 12 --time-budget-min {budget}

import pandas as pd
cols = ["garment_miou", "miou", "iou_upper-clothes", "iou_pants", "iou_skirt", "iou_dress", "iou_belt", "iou_scarf", "val_loss", "train_min"]
base = pd.read_csv(f"{DRIVE_ROOT}/runs/r1_segformer_b3/val_log.csv").set_index("epoch")
assert os.path.exists(f"{DRIVE_ROOT}/runs/{EDGE_RUN}/val_log.csv"), "The edge run has not finished an epoch yet. Read the training output above for the reason."
edge = pd.read_csv(f"{DRIVE_ROOT}/runs/{EDGE_RUN}/val_log.csv").set_index("epoch")
print("\\ngarment mIoU per epoch")
print(pd.DataFrame({"baseline B3": base["garment_miou"], "B3 + edge": edge["garment_miou"]}).to_string())
last = min(len(base), len(edge))
print(f"\\nepoch {last}")
print(pd.DataFrame({"baseline B3": base.loc[last, cols], "B3 + edge": edge.loc[last, cols],
                    "difference": edge.loc[last, cols] - base.loc[last, cols]}).round(4).to_string())'''),
    md("""## 8. Final training run

Trains the chosen model from the start with a longer schedule, as a new run. It takes several hours. If the session ends, run cells 1 to 4 and this cell again: it continues from the last checkpoint.

Set `FINAL_MODEL` to `"segformer_b3_edge"` only if we agreed that the edge model won the experiment above.

When it has finished, set `RUN_NAME` in cell 1 to the name printed below and run cells 5, 6 and 9 for the final model."""),
    code('''FINAL_MODEL, FINAL_EPOCHS = "segformer_b3", 30
FINAL_RUN = f"final_{FINAL_MODEL}_e{FINAL_EPOCHS}"
print("run name:", FINAL_RUN)
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 4 and this cell again.")

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/train.py --drive-root "{DRIVE_ROOT}" --run-name {FINAL_RUN} --model {FINAL_MODEL} --height {HEIGHT} --width {WIDTH} --epochs {FINAL_EPOCHS} --time-budget-min {budget}'''),
    md("""## 9. Final test-set evaluation

**Run this once, only for the final model.** The test split must not be used to choose between models; if it is, its numbers are no longer an honest estimate."""),
    code('''I_CONFIRM_THIS_IS_THE_FINAL_MODEL = False

assert I_CONFIRM_THIS_IS_THE_FINAL_MODEL, "Set the flag to True only when the model and its settings are frozen."
OUT = f"{DRIVE_ROOT}/reports/eval_{RUN_NAME}_test"
!python scripts/evaluate.py --checkpoint "{CHECKPOINT}" --split test --out-dir "{OUT}"
for name in ("confusion_matrix.png", "worst_examples.png", "best_examples.png"):
    display(Image(f"{OUT}/{name}"))'''),
    md("""## What to send back

The text output of cell 5 and a note on how the overlays in cell 6 look (which garments are right, which are wrong)."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.validate(nb)
nbf.write(nb, "notebooks/colab_eval.ipynb")

# ---- A small notebook for the augmentation ablation; it fits a 16 GB T4. ----
abl_cells = [
    md("""# Clothes segmentation: does data augmentation help?

Trains SegFormer-B2 for 5 epochs **without** data augmentation and compares it with the round 1 run `r1_segformer_b2`, which used augmentation and is identical otherwise (same data, seed, batch size and 12-epoch schedule).

SegFormer-B2 is used because it fits a 16 GB T4 at batch size 16. Expect roughly 2 to 3 hours on a T4. If the session ends, run all cells again: training continues from the last checkpoint."""),
    md("## 1. Settings"),
    code(f'''REPO_URL = ""

DRIVE_ROOT = "/content/drive/MyDrive/clothseg"
CODE_DIR = "/content/clothes-segmentation"
CODE_VERSION = "{VERSION}"
HEIGHT, WIDTH = 576, 384
SESSION_BUDGET_MIN = 280        # stop cleanly and save before a 5-hour session ends'''),
    md("## 2. Mount Drive and show the environment"),
    code(train_cell("drive.mount")),
    md("## 3. Get the code"),
    code(train_cell("def code_version")),
    md("## 4. Install requirements and prepare the dataset"),
    code(train_cell("pip install") + '\n!python scripts/prepare_data.py --drive-root "{DRIVE_ROOT}" --height {HEIGHT} --width {WIDTH}'),
    md("## 5. Train without augmentation and compare"),
    code('''RUN, BASE = "abl_segformer_b2_noaug", "r1_segformer_b2"

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/train.py --drive-root "{DRIVE_ROOT}" --run-name {RUN} --model segformer_b2 --height {HEIGHT} --width {WIDTH} --epochs 12 --stop-after-epoch 5 --no-augment --time-budget-min {budget}

import pandas as pd
assert os.path.exists(f"{DRIVE_ROOT}/runs/{RUN}/val_log.csv"), "No epoch has finished yet. Read the training output above for the reason."
base = pd.read_csv(f"{DRIVE_ROOT}/runs/{BASE}/val_log.csv").set_index("epoch")
noaug = pd.read_csv(f"{DRIVE_ROOT}/runs/{RUN}/val_log.csv").set_index("epoch")
print("\\ngarment mIoU and validation loss per epoch")
print(pd.DataFrame({"with augmentation": base["garment_miou"], "without": noaug["garment_miou"],
                    "val loss with": base["val_loss"], "val loss without": noaug["val_loss"]}).dropna().round(4).to_string())
train_loss = lambda name: pd.read_csv(f"{DRIVE_ROOT}/runs/{name}/train_log.csv").groupby("epoch")["loss"].mean()
print("\\nmean training loss per epoch (a low training loss with a high validation loss means overfitting)")
print(pd.DataFrame({"with augmentation": train_loss(BASE), "without": train_loss(RUN)}).dropna().round(4).to_string())'''),
    md("""## What to send back

The whole text output of cell 5."""),
]
abl = nbf.v4.new_notebook(cells=abl_cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.validate(abl)
nbf.write(abl, "notebooks/colab_ablation_augmentation.ipynb")

# ---- Two improvement experiments on the final model: multi-scale averaging, and better starting weights. ----
imp_cells = [
    md("""# Clothes segmentation: two improvement experiments

**A. Multi-scale averaging (cell 5).** No training. The final model predicts each photo at three sizes and mirrored, and the answers are averaged. About 15 minutes. Works on a T4.

**B. Better starting weights (cell 6). Optional, and not recommended as the next step** (PLAN.md, section 25). Trains SegFormer-B3 for 12 epochs starting from weights that were already trained for segmentation on ADE20K, and compares it with the 12-epoch model that started from ImageNet weights. About 2.5 hours. Needs the L4. A better start mainly speeds up early training, so a lead at epoch 12 would not prove a better final model.

Both decisions are made on the validation split, by rules written down in PLAN.md, section 24, before the results exist."""),
    md("## 1. Settings"),
    code(f'''REPO_URL = ""

DRIVE_ROOT = "/content/drive/MyDrive/clothseg"
CODE_DIR = "/content/clothes-segmentation"
CODE_VERSION = "{VERSION}"
HEIGHT, WIDTH = 576, 384
SESSION_BUDGET_MIN = 600'''),
    md("## 2. Mount Drive and show the environment"),
    code(train_cell("drive.mount")),
    md("## 3. Get the code"),
    code(train_cell("def code_version")),
    md("## 4. Install requirements and prepare the dataset"),
    code(train_cell("pip install") + '\n!python scripts/prepare_data.py --drive-root "{DRIVE_ROOT}" --height {HEIGHT} --width {WIDTH}'),
    md("## 5. Experiment A: multi-scale averaging on the final model"),
    code('''import json, subprocess, sys

RUN_NAME = "final_segformer_b3_e30"
CHECKPOINT = f"{DRIVE_ROOT}/runs/{RUN_NAME}/checkpoints/best.pt"
SCALES = ["0.75", "1.0", "1.25"]
MIN_GAIN = 0.002      # multi-scale is adopted if it adds at least 0.2 points of validation mIoU over flip alone

os.chdir(CODE_DIR)
assert os.path.exists(CHECKPOINT), f"No checkpoint at {CHECKPOINT}."

def evaluate(split, tag, extra):
    out = f"{DRIVE_ROOT}/reports/eval_{RUN_NAME}_{split}_{tag}"
    print(f"evaluating {split}, {tag} ...", flush=True)
    r = subprocess.run([sys.executable, "scripts/evaluate.py", "--checkpoint", CHECKPOINT, "--split", split, "--out-dir", out] + extra,
                       capture_output=True, text=True)
    assert r.returncode == 0, "The evaluation failed. Send me this message: " + (r.stdout + r.stderr)[-3000:]
    return json.load(open(f"{out}/metrics.json"))

KEYS = ("miou", "garment_miou", "core_garment_miou", "clothing_binary_iou", "pixel_acc", "boundary_miou", "garment_boundary_miou", "ms_per_image")

def table(title, a, b):
    print()
    print(f"{title}  ({a['num_images']} images)")
    print(f"{'metric':<24}{'flip only':>11}{'multi-scale':>13}{'change':>9}")
    for k in KEYS:
        print(f"{k:<24}{a[k]:>11.4f}{b[k]:>13.4f}{b[k] - a[k]:>+9.4f}")

val_flip = evaluate("val", "flip2", ["--flip-tta"])
val_ms = evaluate("val", "multiscale", ["--flip-tta", "--scales"] + SCALES)
table("VALIDATION", val_flip, val_ms)

# Diagnostic: each input size on its own (with flip). It shows whether a larger input helps the
# thin classes, which decides whether training at a higher resolution is worth trying.
import pandas as pd
WATCH = ["belt", "scarf", "sunglasses", "shoes", "skirt", "dress", "upper-clothes", "pants"]
rows = {}
for tag, extra in (("size x0.75", ["--flip-tta", "--scales", "0.75"]), ("size x1.0", None),
                   ("size x1.25", ["--flip-tta", "--scales", "1.25"]), ("size x1.5", ["--flip-tta", "--scales", "1.5"]),
                   ("x0.75+1+1.25", None)):
    name = {"size x1.0": "flip2", "x0.75+1+1.25": "multiscale"}.get(tag) or "scale_" + extra[-1].replace(".", "p")
    m = evaluate("val", name, extra) if extra else json.load(open(f"{DRIVE_ROOT}/reports/eval_{RUN_NAME}_val_{name}/metrics.json"))
    pc = pd.read_csv(f"{DRIVE_ROOT}/reports/eval_{RUN_NAME}_val_{name}/per_class.csv").set_index("class")["iou"]
    rows[tag] = {"mIoU": m["miou"], "garment": m["garment_miou"], **{c: pc[c] for c in WATCH}}
print()
print("VALIDATION, IoU by input size (all with flip averaging)")
print(pd.DataFrame(rows).T.round(4).to_string())

gain = val_ms["miou"] - val_flip["miou"]
adopted = gain >= MIN_GAIN
print()
print(f"multi-scale changes validation mIoU by {100 * gain:+.2f} points -> " + ("ADOPTED" if adopted else "NOT adopted") + " (rule: at least +0.20)")

if adopted:
    # The test split was already evaluated once with flip averaging. This is a second evaluation,
    # for a setting that was chosen on validation only.
    test_flip = json.load(open(f"{DRIVE_ROOT}/reports/eval_{RUN_NAME}_test_flip/metrics.json"))
    test_ms = evaluate("test", "multiscale", ["--flip-tta", "--scales"] + SCALES)
    table("TEST", test_flip, test_ms)
else:
    print("The test split is not evaluated again. The reported result stays as it is.")'''),
    md("## 6. Experiment B: SegFormer-B3 started from ADE20K segmentation weights"),
    code('''ADE_RUN, BASE_RUN = "abl_segformer_b3_ade", "r1_segformer_b3"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 4 and this cell again.")

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/train.py --drive-root "{DRIVE_ROOT}" --run-name {ADE_RUN} --model segformer_b3_ade --height {HEIGHT} --width {WIDTH} --epochs 12 --time-budget-min {budget}

import pandas as pd
assert os.path.exists(f"{DRIVE_ROOT}/runs/{ADE_RUN}/val_log.csv"), "No epoch has finished yet. Read the training output above for the reason."
cols = ["garment_miou", "miou", "iou_upper-clothes", "iou_pants", "iou_skirt", "iou_dress", "iou_belt", "iou_scarf", "val_loss", "train_min"]
base = pd.read_csv(f"{DRIVE_ROOT}/runs/{BASE_RUN}/val_log.csv").set_index("epoch")
ade = pd.read_csv(f"{DRIVE_ROOT}/runs/{ADE_RUN}/val_log.csv").set_index("epoch")
print()
print("garment mIoU per epoch")
print(pd.DataFrame({"ImageNet start": base["garment_miou"], "ADE20K start": ade["garment_miou"]}).to_string())
last = min(len(base), len(ade))
print()
print(f"epoch {last}")
print(pd.DataFrame({"ImageNet start": base.loc[last, cols], "ADE20K start": ade.loc[last, cols],
                    "difference": ade.loc[last, cols] - base.loc[last, cols]}).round(4).to_string())'''),
    md("""## 7. Helper for comparing a new model with the current best one

Run this once before cells 8 and 9. It evaluates a checkpoint on the validation split at the original image size with flip averaging, and prints it next to the current best model."""),
    code('''import json, subprocess, sys
import pandas as pd

BASE_RUN = "final_segformer_b3_e30"
BASE_CKPT = f"{DRIVE_ROOT}/runs/{BASE_RUN}/checkpoints/best.pt"
CORE = ["upper-clothes", "pants", "skirt", "dress"]
WATCH = CORE + ["belt", "scarf", "sunglasses", "shoes"]
assert os.path.exists(BASE_CKPT), f"The current best model is missing: {BASE_CKPT}"

def val_eval(checkpoint, tag):
    out = f"{DRIVE_ROOT}/reports/eval_{tag}_val_flip"
    print(f"evaluating {tag} on validation ...", flush=True)
    r = subprocess.run([sys.executable, "scripts/evaluate.py", "--checkpoint", checkpoint, "--split", "val", "--out-dir", out, "--flip-tta"],
                       capture_output=True, text=True, cwd=CODE_DIR)
    assert r.returncode == 0, "The evaluation failed. Send me this message: " + (r.stdout + r.stderr)[-3000:]
    return json.load(open(f"{out}/metrics.json")), pd.read_csv(f"{out}/per_class.csv").set_index("class")["iou"]

def compare(checkpoint, tag):
    """Print the new model next to the current best one and say whether it meets the rule fixed in PLAN.md, section 26."""
    assert os.path.exists(checkpoint), f"No checkpoint at {checkpoint}. Did the training cell finish?"
    base_m, base_pc = val_eval(BASE_CKPT, BASE_RUN)
    new_m, new_pc = val_eval(checkpoint, tag)
    rows = {k: (base_m[k], new_m[k]) for k in ("miou", "garment_miou", "core_garment_miou", "clothing_binary_iou", "boundary_miou")}
    rows.update({c: (base_pc[c], new_pc[c]) for c in WATCH})
    df = pd.DataFrame(rows, index=["current best", tag]).T
    df["change"] = df[tag] - df["current best"]
    print()
    print(df.round(4).to_string())
    gain = new_m["garment_miou"] - base_m["garment_miou"]
    worst_core = min(new_pc[c] - base_pc[c] for c in CORE)
    ok = gain >= 0.005 and worst_core >= -0.01
    print()
    print(f"garment mIoU change {100 * gain:+.2f} points; largest drop on a core garment {100 * worst_core:+.2f} points -> "
          + ("MEETS the rule" if ok else "does NOT meet the rule") + " (at least +0.50, and no core garment below -1.00)")

print("helper ready")'''),
    md("""## 8. Rank 2: short fine-tune with weight averaging

Continues from the current best model in 3 cycles of 3 epochs. Each cycle starts at a quarter of the original learning rate and decays; the weights at the end of each cycle are averaged together with the starting weights. About 2 hours on the L4. The current best model is only read, never changed. If the session ends, run cells 1 to 4, 7 and this cell again."""),
    code('''SWA_RUN = "swa_segformer_b3"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 4, 7 and this cell again.")

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/finetune.py --drive-root "{DRIVE_ROOT}" --init-checkpoint "{BASE_CKPT}" --run-name {SWA_RUN} --cycles 3 --epochs-per-cycle 3 --time-budget-min {budget}

compare(f"{DRIVE_ROOT}/runs/{SWA_RUN}/checkpoints/best.pt", SWA_RUN)'''),
    md("""## 9. Rank 3: fine-tune at a larger input (768×512)

Run this only if the table "IoU by input size" from cell 5 shows the thin classes (belt, scarf, sunglasses) improving at size ×1.25 or ×1.5. It first builds the dataset at 768×512 (several minutes, once), then fine-tunes the current best model for 3 epochs at batch size 8. About 1.5 hours on the L4."""),
    code('''BIG_RUN = "big_segformer_b3_768"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 4, 7 and this cell again.")

os.chdir(CODE_DIR)
!python scripts/prepare_data.py --drive-root "{DRIVE_ROOT}" --height 768 --width 512 --workers 8
budget = int(minutes_left())
!python scripts/finetune.py --drive-root "{DRIVE_ROOT}" --init-checkpoint "{BASE_CKPT}" --run-name {BIG_RUN} --height 768 --width 512 --batch-size 8 --cycles 1 --epochs-per-cycle 3 --no-average --time-budget-min {budget}

compare(f"{DRIVE_ROOT}/runs/{BIG_RUN}/checkpoints/best.pt", BIG_RUN)'''),
    md("""## What to send back

Everything cell 5 prints, and everything cells 8 and 9 print if you run them. The tables at the end of each are the important part. Cell 6 is optional and not recommended."""),
]
# Experiments A and B (cells 5 and 6 above) are finished and were not adopted, so the notebook that
# is written holds only what is still to run: the setup cells, the helper, rank 2 and rank 3.
RENAME = (("## 7. Helper", "## 5. Helper"), ("## 8. Rank 2", "## 6. Rank 2"), ("## 9. Rank 3", "## 7. Rank 3"),
          ("before cells 8 and 9", "before cells 6 and 7"), ("cells 1 to 4, 7 and this cell", "cells 1 to 5 and this cell"),
          ('Run this only if the table "IoU by input size" from cell 5 shows the thin classes (belt, scarf, sunglasses) improving at size ×1.25 or ×1.5. It first',
           "It first"))


def renamed(cell):
    for old, new in RENAME:
        cell["source"] = cell["source"].replace(old, new)
    return cell


ranks_cells = [
    md("""# Clothes segmentation: two fine-tuning experiments on the final model

Both start from the current best model, `final_segformer_b3_e30/best.pt`, which is only read and never changed. Both are judged on the validation split by rules fixed in PLAN.md, section 26.

- **Cell 6, rank 2:** a short fine-tune with weight averaging. About 2 hours.
- **Cell 7, rank 3:** a fine-tune at a larger input, 768×512. About 1.5 hours.

Use the **L4 GPU**. Run the cells from top to bottom; cell 3 asks for `clothseg_code.zip` once. If the session ends, run cells 1 to 5 and the interrupted cell again: it continues from the last finished epoch."""),
] + imp_cells[1:9] + [renamed(c) for c in imp_cells[13:19]] + [
    md("""## 8. Do two models have different strengths? (ensemble test)

No training. It averages the predictions of the final model with those of other trained models and evaluates on the validation split. A clear gain means the models are right in different places, which is the condition for a merged architecture to be worth building. About 25 minutes. Only cells 1 to 4 are needed before it."""),
    code('''import json, os, subprocess, sys
import pandas as pd

os.chdir(CODE_DIR)
R = f"{DRIVE_ROOT}/runs"
CKPT = {
    "final B3": f"{R}/final_segformer_b3_e30/checkpoints/best.pt",
    "UPerNet": f"{R}/r1_upernet_convnext_tiny/checkpoints/best.pt",
    "B4": f"{R}/r1_segformer_b4/checkpoints/best.pt",
    "B3 lr x2": f"{R}/abl_segformer_b3_lr_double/checkpoints/best.pt",
}
COMBOS = [["final B3"], ["UPerNet"], ["B4"], ["B3 lr x2"],
          ["final B3", "UPerNet"], ["final B3", "B4"], ["final B3", "B3 lr x2"], ["final B3", "UPerNet", "B4"]]
WATCH = ["upper-clothes", "pants", "skirt", "dress", "belt", "scarf", "sunglasses"]
missing = [k for k, v in CKPT.items() if not os.path.exists(v)]
COMBOS = [c for c in COMBOS if not set(c) & set(missing)]
if missing:
    print("not found on Drive, left out:", missing)

rows = {}
for combo in COMBOS:
    name = " + ".join(combo)
    out = f"{DRIVE_ROOT}/reports/ens_" + "_".join(k.replace(" ", "") for k in combo)
    print("evaluating", name, "...", flush=True)
    cmd = [sys.executable, "scripts/evaluate.py", "--checkpoint", CKPT[combo[0]], "--split", "val", "--out-dir", out, "--flip-tta"]
    if len(combo) > 1:
        cmd += ["--ensemble-with"] + [CKPT[k] for k in combo[1:]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, "The evaluation failed. Send me this message: " + (r.stdout + r.stderr)[-3000:]
    m = json.load(open(f"{out}/metrics.json"))
    pc = pd.read_csv(f"{out}/per_class.csv").set_index("class")["iou"]
    rows[name] = {"mIoU": m["miou"], "garment": m["garment_miou"], "core": m["core_garment_miou"], "boundary": m["boundary_miou"],
                  **{c: pc[c] for c in WATCH}, "ms/img": m["ms_per_image"]}

df = pd.DataFrame(rows).T
print()
print("VALIDATION, original resolution, flip averaging")
print(df.round(4).to_string())
base = df.loc["final B3"]
print()
for name in df.index:
    if "+" in name:
        gain = df.loc[name, "mIoU"] - base["mIoU"]
        print(f"{name}: mIoU {100 * gain:+.2f} points, garment mIoU {100 * (df.loc[name, 'garment'] - base['garment']):+.2f} points over the final model alone -> "
              + ("CLEAR gain" if gain >= 0.005 else "no clear gain") + " (threshold +0.50)")'''),
    md("""## 9. Merged architecture: SegFormer-B3 encoder + ConvNeXt-Tiny encoder with a fusion decoder

One network with two encoders. Both read the photo; their features are joined at each of the four scales; a new decoder predicts the labels. Trained for 12 epochs with the same data, seed and settings as the single models from the comparison, then printed next to them. About 3 hours on the L4. Only cells 1 to 4 are needed before it. If the session ends, run cells 1 to 4 and this cell again."""),
    code('''import os
import pandas as pd

DUAL_RUN = "abl_dual_b3_convnext_tiny"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 4 and this cell again.")

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/train.py --drive-root "{DRIVE_ROOT}" --run-name {DUAL_RUN} --model dual_b3_convnext_tiny --height {HEIGHT} --width {WIDTH} --epochs 12 --time-budget-min {budget}

COLS = ["garment_miou", "miou", "iou_upper-clothes", "iou_pants", "iou_skirt", "iou_dress", "iou_belt", "iou_scarf", "iou_sunglasses", "val_loss", "train_min"]
CORE = ["iou_upper-clothes", "iou_pants", "iou_skirt", "iou_dress"]
RUNS = {"SegFormer-B3": "r1_segformer_b3", "UPerNet+ConvNeXt-T": "r1_upernet_convnext_tiny", "merged": DUAL_RUN}
logs = {k: pd.read_csv(f"{DRIVE_ROOT}/runs/{v}/val_log.csv").set_index("epoch") for k, v in RUNS.items()
        if os.path.exists(f"{DRIVE_ROOT}/runs/{v}/val_log.csv")}
assert "merged" in logs, "The merged model has not finished an epoch yet. Read the training output above for the reason."
print()
print("garment mIoU per epoch")
print(pd.DataFrame({k: v["garment_miou"] for k, v in logs.items()}).round(4).to_string())
done = len(logs["merged"])
if done < 12:
    print()
    print(f"NOT FINISHED: {done} of 12 epochs. Run this cell again to continue.")
else:
    print()
    print("validation at epoch 12")
    print(pd.DataFrame({k: v.loc[12, COLS] for k, v in logs.items()}).round(4).to_string())
    base, new = logs["SegFormer-B3"].loc[12], logs["merged"].loc[12]
    gain = new["garment_miou"] - base["garment_miou"]
    worst = min(new[c] - base[c] for c in CORE)
    ok = gain >= 0.005 and worst >= -0.01
    print()
    print(f"merged against SegFormer-B3: garment mIoU {100 * gain:+.2f} points; largest drop on a core garment {100 * worst:+.2f} points -> "
          + ("MEETS the rule" if ok else "does NOT meet the rule") + " (at least +0.50, and no core garment below -1.00)")'''),
    md("""## What to send back

Everything cells 6, 7, 8 and 9 print. The table and the last lines of each are the important part."""),
]
ranks = nbf.v4.new_notebook(cells=ranks_cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.validate(ranks)
nbf.write(ranks, "notebooks/colab_ranks.ipynb")

# The merged two-encoder run alone, so it can be started without the ranks notebook.
dual_md, dual_code = ranks_cells[-3], ranks_cells[-2]
dual_cells = [
    md("""# Clothes segmentation: the merged two-encoder model

Trains `dual_b3_convnext_tiny` (SegFormer-B3 encoder + ConvNeXt-Tiny encoder, fusion decoder) for 12 epochs and compares it with the single models. The rule is fixed in PLAN.md, section 32.3.

Use the **L4 GPU**. Run the cells from top to bottom; cell 3 asks for `clothseg_code.zip` once. If the session ends, run all cells again: the training continues from the last finished epoch saved on Drive."""),
] + imp_cells[1:9] + [
    md(dual_md["source"].replace("## 9. Merged", "## 5. Merged")),
    code(dual_code["source"]),
    md("""## 6. Optional: continue the merged model from epoch 12, check at epoch 20, then possibly to 30

The 12-epoch run did not meet its rule (PLAN.md, section 32.6). This continues that finished run, `abl_dual_b3_convnext_tiny`, as a new run on the 30-epoch learning-rate schedule of the final SegFormer-B3 model. Weights, optimizer state and data order carry over; the 12-epoch run itself is only read.

1. With `STOP_AT = 20` it trains epochs 13 to 20 (about 2.5 hours on the L4) and compares epochs 18 to 20 with the same epochs of the 30-epoch SegFormer-B3 run. It prints CONTINUE or STOP (rule in PLAN.md, section 38).
2. Only on CONTINUE: set `STOP_AT = 30` and run cells 1 to 4 and this cell again. It trains epochs 21 to 30 (about 3 hours), then compares both models on the validation split at the original image size with flip averaging (rule in PLAN.md, section 35.2).

Needs code version 14. Only cells 1 to 4 are needed before it. If the session ends, run cells 1 to 4 and this cell again: it continues from its own last checkpoint."""),
    code('''import json, os, subprocess, sys
import pandas as pd

STOP_AT = 20     # first 20; set to 30 and run again only if the epoch-20 check below says CONTINUE
SHORT_RUN = "abl_dual_b3_convnext_tiny"        # finished 12-epoch run, only read
LONG_RUN = "ext_dual_b3_convnext_tiny_e30"      # epochs 13 to 30, on the 30-epoch schedule
BASE_RUN = "final_segformer_b3_e30"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 4 and this cell again.")

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/train.py --drive-root "{DRIVE_ROOT}" --run-name {LONG_RUN} --extend-from {SHORT_RUN} --model dual_b3_convnext_tiny --height {HEIGHT} --width {WIDTH} --epochs 30 --stop-after-epoch {STOP_AT} --time-budget-min {budget}

assert os.path.exists(f"{DRIVE_ROOT}/runs/{LONG_RUN}/val_log.csv"), "The continuation has not finished an epoch yet. Read the training output above for the reason."
base = pd.read_csv(f"{DRIVE_ROOT}/runs/{BASE_RUN}/val_log.csv").set_index("epoch")
merged = pd.concat([pd.read_csv(f"{DRIVE_ROOT}/runs/{SHORT_RUN}/val_log.csv").set_index("epoch"),
                    pd.read_csv(f"{DRIVE_ROOT}/runs/{LONG_RUN}/val_log.csv").set_index("epoch")])
done = int(merged.index.max())
print()
print("garment mIoU per epoch, 576x384. Merged: epochs 1-12 from the 12-epoch run, 13 onwards on the 30-epoch schedule")
print(pd.DataFrame({"SegFormer-B3 (30 ep)": base["garment_miou"], "merged": merged["garment_miou"]}).round(4).to_string())
CORE = ["upper-clothes", "pants", "skirt", "dress"]

if done < 20:
    print()
    print(f"NOT FINISHED: {done} of 20 epochs. Run cells 1 to 4 and this cell again to continue.")
elif done < 30:
    # Check at epoch 20, fixed before the result (PLAN.md section 38): both models are on the same
    # 30-epoch schedule here, so their epochs 18-20 are compared directly. Three epochs are averaged
    # because single epochs move by about 0.5 points.
    m, b = merged.loc[18:20], base.loc[18:20]
    cols = ["garment_miou", "miou"] + [f"iou_{c}" for c in CORE] + ["iou_belt", "iou_scarf", "iou_sunglasses"]
    table = pd.DataFrame({"SegFormer-B3": b[cols].mean(), "merged": m[cols].mean()})
    table["difference"] = table["merged"] - table["SegFormer-B3"]
    print()
    print("EPOCH-20 CHECK: mean of epochs 18-20, validation 576x384")
    print(table.round(4).to_string())
    gain = table.loc["garment_miou", "difference"]
    worst = min(table.loc[f"iou_{c}", "difference"] for c in CORE)
    go = gain >= 0.005 and worst >= -0.01
    print()
    print(f"garment mIoU {100 * gain:+.2f} points; largest drop on a core garment {100 * worst:+.2f} points -> "
          + ("CONTINUE: set STOP_AT = 30 and run cells 1 to 4 and this cell again" if go else "STOP here: the merged model is not adopted")
          + " (rule: at least +0.50, and no core garment below -1.00)")
else:
    WATCH = CORE + ["belt", "scarf", "sunglasses", "shoes"]
    rows = {}
    for name, run in (("SegFormer-B3", BASE_RUN), ("merged", LONG_RUN)):
        out = f"{DRIVE_ROOT}/reports/eval_{run}_val_flip"
        print("evaluating", name, "on validation, original size, flip ...", flush=True)
        r = subprocess.run([sys.executable, "scripts/evaluate.py", "--checkpoint", f"{DRIVE_ROOT}/runs/{run}/checkpoints/best.pt",
                            "--split", "val", "--out-dir", out, "--flip-tta"], capture_output=True, text=True)
        assert r.returncode == 0, "The evaluation failed. Send me this message: " + (r.stdout + r.stderr)[-3000:]
        mt = json.load(open(f"{out}/metrics.json"))
        pc = pd.read_csv(f"{out}/per_class.csv").set_index("class")["iou"]
        rows[name] = {"garment_miou": mt["garment_miou"], "miou": mt["miou"], "core_garment_miou": mt["core_garment_miou"],
                      "boundary_miou": mt["boundary_miou"], **{c: pc[c] for c in WATCH}, "ms_per_image": mt["ms_per_image"]}
    df = pd.DataFrame(rows)
    df["difference"] = df["merged"] - df["SegFormer-B3"]
    print()
    print("VALIDATION, original resolution, flip averaging, best.pt of each run")
    print(df.round(4).to_string())
    gain = df.loc["garment_miou", "difference"]
    worst = min(df.loc[c, "difference"] for c in CORE)
    ok = df.loc["garment_miou", "merged"] >= 0.7491 and worst >= -0.01
    print()
    print(f"merged against SegFormer-B3: garment mIoU {100 * gain:+.2f} points; largest drop on a core garment {100 * worst:+.2f} points -> "
          + ("MEETS the rule" if ok else "does NOT meet the rule") + " (garment mIoU at least 0.7491, and no core garment below -1.00)")'''),
    md("""## 7. Do SegFormer-B3 and the ConvNeXt model make different mistakes? (ensemble test)

No training. Scores the final SegFormer-B3 alone, UPerNet + ConvNeXt-T alone, and the two with their class probabilities averaged pixel by pixel, on the validation split at the original size with flip averaging. If the average is clearly better, the ConvNeXt model knows something about garments that SegFormer-B3 does not, and a better fusion design could use it; if not, no fusion design can add much. Rule in PLAN.md, section 39.4. About 10 minutes; a T4 is enough. Only cells 1 to 4 are needed before it; run it when cell 6 is not training."""),
    code('''import json, os, subprocess, sys
import pandas as pd

os.chdir(CODE_DIR)
R = f"{DRIVE_ROOT}/runs"
B3 = f"{R}/final_segformer_b3_e30/checkpoints/best.pt"
UPER = f"{R}/r1_upernet_convnext_tiny/checkpoints/best.pt"
CORE = ["upper-clothes", "pants", "skirt", "dress"]
WATCH = CORE + ["belt", "scarf", "sunglasses", "shoes"]

rows = {}
for name, ckpts in (("B3 alone", [B3]), ("UPerNet alone", [UPER]), ("B3 + UPerNet averaged", [B3, UPER])):
    out = f"{DRIVE_ROOT}/reports/ens_test_" + str(len(rows))
    print("evaluating", name, "...", flush=True)
    cmd = [sys.executable, "scripts/evaluate.py", "--checkpoint", ckpts[0], "--split", "val", "--out-dir", out, "--flip-tta"]
    if len(ckpts) > 1:
        cmd += ["--ensemble-with"] + ckpts[1:]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, "The evaluation failed. Send me this message: " + (r.stdout + r.stderr)[-3000:]
    m = json.load(open(f"{out}/metrics.json"))
    pc = pd.read_csv(f"{out}/per_class.csv").set_index("class")["iou"]
    rows[name] = {"garment_miou": m["garment_miou"], "miou": m["miou"], "boundary_miou": m["boundary_miou"],
                  **{c: pc[c] for c in WATCH}}

df = pd.DataFrame(rows)
df["average minus B3"] = df["B3 + UPerNet averaged"] - df["B3 alone"]
print()
print("VALIDATION, original resolution, flip averaging")
print(df.round(4).to_string())
gain = df.loc["garment_miou", "average minus B3"]
print()
print(f"garment mIoU of the average over B3 alone: {100 * gain:+.2f} points -> "
      + ("the encoders carry DIFFERENT garment information (fusion is wasting it)" if gain >= 0.005
         else "the encoders carry little different garment information (no fusion design can add much)")
      + " (threshold +0.50, PLAN.md section 39.4)")'''),
    md("""## What to send back

Everything cell 5 prints, and everything cells 6 and 7 print if you run them. The first lines (memory and time per epoch) and the tables at the end are the important part."""),
]
dual_nb = nbf.v4.new_notebook(cells=dual_cells, metadata=ranks.metadata)
nbf.validate(dual_nb)
nbf.write(dual_nb, "notebooks/colab_dual.ipynb")

# ---- Segment your own photos: upload, see every output, upload again. Nothing is trained. ----
photo_cells = [
    md("""# Clothes segmentation: try the model on your own photos

Upload a photo of a person; the final model segments it and shows:

1. the original, 2. the segmentation in colours (all 15 classes), 3. the colours over the photo,
4. **only the clothes, everything else black**, 5. the clothes as a white mask,
6. the clothes in their class colours, everything else black, 7. the person with the background removed,
8. only the body (hair, face, arms, legs), and then **each garment found, alone on black**.

Every picture is also saved to Drive (`clothseg/reports/my_photos/<photo name>/`), together with `clothes_transparent.png` (the clothes on a see-through background) and `label_mask.png`.

After each photo the notebook asks whether you want to upload another one.

A **T4 GPU is enough**. Run the cells from top to bottom; cell 3 asks for `clothseg_code.zip` only if Drive has an older code version. "Clothes" means the 10 clothing and accessory classes: hat, sunglasses, upper-clothes, skirt, pants, dress, belt, shoes, bag, scarf."""),
    md("## 1. Settings"),
    code('''REPO_URL = ""

DRIVE_ROOT = "/content/drive/MyDrive/clothseg"
CODE_DIR = "/content/clothes-segmentation"
CODE_VERSION = "@VERSION@"
SESSION_BUDGET_MIN = 600   # used by a helper in cell 2; nothing is trained here

RUN_NAME = "final_segformer_b3_e30"   # the final model
CHECKPOINT = f"{DRIVE_ROOT}/runs/{RUN_NAME}/checkpoints/best.pt"
FLIP = True        # flip averaging, as in the reported results; False is about twice as fast
MIN_AREA = 0.002   # a garment smaller than 0.2% of the photo is not shown on its own
DOWNLOAD = False   # True: also download a zip of each photo's pictures to your computer'''.replace("@VERSION@", VERSION)),
    md("## 2. Mount Drive and show the environment"),
    code(train_cell("drive.mount")),
    md("## 3. Get the code"),
    code(train_cell("def code_version")),
    md("## 4. Install requirements"),
    code(train_cell("pip install")),
    md("## 5. Load the model"),
    code('''import sys
sys.path.insert(0, f"{CODE_DIR}/src")
from clothseg.inference import Segmenter, load_image
from clothseg.labels import garment_ids

assert os.path.exists(CHECKPOINT), f"Model not found: {CHECKPOINT}. Check RUN_NAME in cell 1."
seg = Segmenter(CHECKPOINT, flip_tta=FLIP)
print(f"model loaded: {seg.cfg.model}, epoch {seg.epoch}, {seg.num_classes} classes | flip averaging {'on' if FLIP else 'off'} | {seg.device}")'''),
    md("## 6. Helper functions (run once)"),
    code('''import io, math, re, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

OUT_ROOT = f"{DRIVE_ROOT}/reports/my_photos"
GARMENTS = garment_ids(seg.num_classes)                    # hat, sunglasses, upper-clothes, skirt, pants, dress, belt, shoes, bag, scarf
BODY = [i for i, n in enumerate(seg.names) if n in ("hair", "face", "arms", "legs")]


def keep(image, region):
    """The photo where region is True, black everywhere else."""
    return np.where(region[..., None], image, 0).astype(np.uint8)


def views(image, mask):
    """All output pictures for one photo, as (title, RGB array) pairs."""
    clothes = np.isin(mask, GARMENTS)
    colours = seg.colorize(mask)
    out = [
        ("1. Original", image),
        ("2. Segmentation, all classes", colours),
        ("3. Overlay on the photo", seg.overlay(image, mask)),
        ("4. Clothes only, rest black", keep(image, clothes)),
        ("5. Clothes mask (white)", np.repeat(clothes[..., None], 3, axis=2).astype(np.uint8) * 255),
        ("6. Clothes in class colours, rest black", keep(colours, clothes)),
        ("7. Person, background removed", keep(image, mask != 0)),
        ("8. Body only: hair, face, arms, legs", keep(image, np.isin(mask, BODY))),
    ]
    for cid in GARMENTS:
        region = mask == cid
        if region.mean() >= MIN_AREA:
            out.append((f"{len(out) + 1}. Only the {seg.names[cid]}", keep(image, region)))
    return out


def show(name, image, mask, pictures):
    """Draw all pictures in a grid with a colour legend of the classes found."""
    cols = 4
    rows = math.ceil(len(pictures) / cols)
    h, w = image.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * (5 * h / w + 0.45) + 1.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, pic) in zip(axes.flat, pictures):
        ax.imshow(pic)
        ax.set_title(title, fontsize=11)
    present = [i for i in np.unique(mask) if i != 0]
    handles = [mpatches.Patch(color=seg.palette[i] / 255, label=seg.names[i]) for i in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, max(1, len(handles))), fontsize=11, frameon=False)
    fig.suptitle(name, fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    plt.show()


def process(name, data):
    """Segment one uploaded photo, show every output, save them to Drive. Returns the output folder."""
    image = load_image(io.BytesIO(data))
    mask = seg.predict(image)
    pictures = views(image, mask)
    show(name, image, mask, pictures)

    counts = np.bincount(mask.ravel(), minlength=seg.num_classes)
    found = sorted(((100 * counts[i] / mask.size, seg.names[i]) for i in range(1, seg.num_classes) if counts[i] >= 0.0005 * mask.size), reverse=True)
    print("share of the photo per class:  " + ",  ".join(f"{n} {p:.1f}%" for p, n in found))
    missing = [seg.names[i] for i in GARMENTS if counts[i] and counts[i] / mask.size < MIN_AREA]
    if missing:
        print(f"not shown on their own (under {100 * MIN_AREA:.1f}% of the photo):", ", ".join(missing))

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name.rsplit(".", 1)[0]).strip("_") or "photo"
    out_dir = f"{OUT_ROOT}/{stem}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    for k, (title, pic) in enumerate(pictures, 1):
        slug = re.sub(r"[^a-z0-9]+", "_", title.split(". ", 1)[-1].lower()).strip("_")
        Image.fromarray(pic).save(f"{out_dir}/{k:02d}_{slug}.png")
    clothes = np.isin(mask, GARMENTS)
    Image.fromarray(np.dstack([image, clothes.astype(np.uint8) * 255])).save(f"{out_dir}/clothes_transparent.png")
    label_png = Image.frombytes("P", (mask.shape[1], mask.shape[0]), np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    label_png.putpalette(seg.palette.flatten().tolist())
    label_png.save(f"{out_dir}/label_mask.png")
    print(f"saved {len(pictures) + 2} pictures to {out_dir} (including clothes_transparent.png, a cut-out with a see-through background)")
    return out_dir'''),
    md("""## 7. Upload photos and see the results

Choose one or more photos in the upload box. After they are shown, type `y` to upload another photo, or anything else to stop. If a photo looks wrong, the reasons the model can fail are listed in the report (report_latex/report.pdf), sections 5.2 and 5.3."""),
    code('''from google.colab import files

done = 0
while True:
    print()
    print("Choose one or more photos of a person (JPG or PNG).")
    try:
        uploaded = files.upload()
    except Exception as exc:  # the upload box was cancelled or the browser lost the connection
        print(f"upload stopped ({type(exc).__name__})")
        uploaded = {}
    for name, data in uploaded.items():
        try:
            out_dir = process(name, data)
        except Exception as exc:
            print(f"{name}: could not be segmented ({type(exc).__name__}: {exc}). Is it an image file?")
            continue
        done += 1
        if DOWNLOAD:
            shutil.make_archive(out_dir, "zip", out_dir)
            files.download(out_dir + ".zip")
    if not uploaded:
        print("no photo was received")
    answer = input("Upload another photo? Type y for yes, anything else to stop: ").strip().lower()
    if answer not in ("y", "yes"):
        break
print(f"Finished. {done} photo(s) segmented. All results are on Drive in {OUT_ROOT}")'''),
]
photos_nb = nbf.v4.new_notebook(cells=photo_cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.validate(photos_nb)
nbf.write(photos_nb, "notebooks/colab_my_photos.ipynb")

# ---- Test on a new, small dataset (CCP): before training, a public model, and our final model. ----
ccp_cells = [
    md("""# Clothes segmentation: test on a new dataset the model has never seen

**Dataset:** Clothing Co-Parsing (CCP), [github.com/bearpaw/clothing-co-parsing](https://github.com/bearpaw/clothing-co-parsing), Apache-2.0 licence. 1,004 street-fashion photos (about 550x830) with pixel labels in 59 classes. It is downloaded here; nothing is trained on it.

**Three models are compared on the same photos:**

1. **Before training:** our SegFormer-B3 exactly as it was before we trained it: ImageNet encoder weights and an untrained decoder. It shows what our training added.
2. **A public model:** `mattmdjaga/segformer_b2_clothes` from Hugging Face, a SegFormer-B2 that someone else trained on ATR. It shows how we compare with what anyone can download.
3. **Our final model:** `final_segformer_b3_e30`, with flip averaging.

**Rules fixed before the result** (PLAN.md, section 41): CCP's 59 labels are mapped to 13 comparison classes (face, arms and legs become one "skin" class, because CCP has only "skin"); labels without a clear match (for example stockings, socks, jewellery) are ignored. CCP photos that are also in ATR, found by image hash, are removed, because ATR was partly built from older fashion datasets.

A **T4 GPU is enough**. About 15 minutes. Run the cells from top to bottom."""),
    md("## 1. Settings"),
    code('''REPO_URL = ""

DRIVE_ROOT = "/content/drive/MyDrive/clothseg"
CODE_DIR = "/content/clothes-segmentation"
CODE_VERSION = "@VERSION@"
HEIGHT, WIDTH = 576, 384
SESSION_BUDGET_MIN = 600   # used by a helper in cell 2; nothing is trained here

ATR_CACHE = f"/content/data/atr_{HEIGHT}x{WIDTH}"   # its meta.csv holds the image hash of every ATR photo
OUR_CHECKPOINT = f"{DRIVE_ROOT}/runs/final_segformer_b3_e30/checkpoints/best.pt"
BEFORE_PATH = "/content/before_training.pt"
PUBLIC_MODEL = "mattmdjaga/segformer_b2_clothes"'''.replace("@VERSION@", VERSION)),
    md("## 2. Mount Drive and show the environment"),
    code(train_cell("drive.mount")),
    md("## 3. Get the code"),
    code(train_cell("def code_version")),
    md("## 4. Install requirements and prepare ATR (needed only for the overlap check)"),
    code(train_cell("pip install") + '\n!python scripts/prepare_data.py --drive-root "{DRIVE_ROOT}" --height {HEIGHT} --width {WIDTH}'),
    md("## 5. Download CCP"),
    code('''import os, subprocess

CCP_DIR = "/content/ccp"
if not os.path.isdir(f"{CCP_DIR}/annotations/pixel-level"):
    print("downloading the Clothing Co-Parsing dataset from GitHub (about 130 MB) ...", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/bearpaw/clothing-co-parsing.git", CCP_DIR],
                       capture_output=True, text=True)
    assert r.returncode == 0, "Download failed. Send me this message: " + r.stderr[-2000:]
n_ann = len([f for f in os.listdir(f"{CCP_DIR}/annotations/pixel-level") if f.endswith(".mat")])
n_img = len([f for f in os.listdir(f"{CCP_DIR}/photos") if f.endswith(".jpg")])
print(f"CCP ready: {n_img} photos, {n_ann} with pixel labels (only these are used)")'''),
    md("## 6. Read the labels, map them, and remove photos that are also in ATR"),
    code('''import glob, os, sys
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

sys.path.insert(0, f"{CODE_DIR}/src")
from clothseg.data.preprocess import _dhash
from clothseg.inference import load_image

# The classes all models are compared on. CCP has one "skin" class, so face, arms and legs
# of the models are merged into "skin" before comparing.
EVAL = ["background", "hat", "hair", "sunglasses", "upper-clothes", "skirt", "pants", "dress", "belt", "shoes", "skin", "bag", "scarf"]
GARMENTS = ["hat", "sunglasses", "upper-clothes", "skirt", "pants", "dress", "belt", "shoes", "bag", "scarf"]
# CCP label -> comparison class. Mapping fixed before any result (PLAN.md section 41); labels
# without a clear match are ignored (left out of the scores), not counted as errors.
CCP_TO_EVAL = {
    "null": "background", "hat": "hat", "hair": "hair", "sunglasses": "sunglasses", "skin": "skin",
    "blazer": "upper-clothes", "blouse": "upper-clothes", "cape": "upper-clothes", "cardigan": "upper-clothes",
    "coat": "upper-clothes", "hoodie": "upper-clothes", "jacket": "upper-clothes", "shirt": "upper-clothes",
    "sweater": "upper-clothes", "sweatshirt": "upper-clothes", "t-shirt": "upper-clothes", "top": "upper-clothes",
    "vest": "upper-clothes",
    "skirt": "skirt", "dress": "dress", "belt": "belt", "scarf": "scarf", "bag": "bag", "purse": "bag",
    "jeans": "pants", "leggings": "pants", "pants": "pants", "shorts": "pants",
    "boots": "shoes", "clogs": "shoes", "flats": "shoes", "heels": "shoes", "loafers": "shoes", "pumps": "shoes",
    "sandals": "shoes", "shoes": "shoes", "sneakers": "shoes", "wedges": "shoes",
}
CCP_IGNORED = {"accessories", "bodysuit", "bra", "bracelet", "earrings", "glasses", "gloves", "intimate", "jumper",
               "necklace", "panties", "ring", "romper", "socks", "stockings", "suit", "swimwear", "tie", "tights",
               "wallet", "watch"}
OVERLAP_MAX_BITS = 4   # a CCP photo whose hash differs from an ATR photo's in at most 4 of 64 bits counts as the same photo

names = [str(a[0]) if a.size else "" for a in sio.loadmat(f"{CCP_DIR}/label_list.mat")["label_list"].ravel()]
unknown = [n for n in names if n not in CCP_TO_EVAL and n not in CCP_IGNORED]
assert not unknown, f"CCP labels without a rule: {unknown}"
ccp_lut = np.full(256, 255, np.uint8)
for i, n in enumerate(names):
    if n in CCP_TO_EVAL:
        ccp_lut[i] = EVAL.index(CCP_TO_EVAL[n])

# Hashes of every ATR photo, computed the same way during preprocessing (on the original photo).
meta = pd.read_csv(f"{ATR_CACHE}/meta.csv", dtype={"dhash": str})
atr_hash = np.array([int(h, 16) for h in meta["dhash"]], dtype=np.uint64)

samples, skipped, ignored_px, total_px = [], [], 0, 0
for path in sorted(glob.glob(f"{CCP_DIR}/annotations/pixel-level/*.mat")):
    sid = os.path.basename(path)[:-4]
    raw = sio.loadmat(path)["groundtruth"]
    photo = f"{CCP_DIR}/photos/{sid}.jpg"
    if not os.path.exists(photo):
        skipped.append((sid, "no photo"))
        continue
    image = load_image(photo)
    if image.shape[:2] != raw.shape:
        skipped.append((sid, f"photo {image.shape[:2]} and labels {raw.shape} differ in size"))
        continue
    gt = ccp_lut[raw]
    ignored_px += int((gt == 255).sum())
    total_px += gt.size
    h = np.uint64(int(_dhash(Image.fromarray(image)), 16))
    bits = int(np.unpackbits((atr_hash ^ h).view(np.uint8)).reshape(-1, 64).sum(1).min())
    samples.append({"id": sid, "photo": photo, "gt": gt, "bits_to_nearest_atr": bits})

dist = np.array([s["bits_to_nearest_atr"] for s in samples])
clean = [s for s in samples if s["bits_to_nearest_atr"] > OVERLAP_MAX_BITS]
print(f"labelled photos read: {len(samples)}; skipped: {len(skipped)} {skipped[:3]}")
print(f"pixels with an ignored CCP label: {100 * ignored_px / max(1, total_px):.2f}%")
print(f"overlap with ATR (64-bit image hash): identical {int((dist == 0).sum())}, within {OVERLAP_MAX_BITS} bits {int((dist <= OVERLAP_MAX_BITS).sum())}, "
      f"within 8 bits {int((dist <= 8).sum())}")
print(f"photos used for the comparison (no overlap with ATR): {len(clean)}")
present = {c: sum(int((s["gt"] == k).any()) for s in clean) for k, c in enumerate(EVAL)}
print("photos containing each class:", ", ".join(f"{c} {n}" for c, n in present.items()))'''),
    md("## 7. Load the three models"),
    code('''import time
import torch
import torch.nn.functional as F
from clothseg.config import TrainConfig
from clothseg.inference import Segmenter
from clothseg.models import build_model


def eval_lut(model_names):
    """Lookup table from a model's class ids to the comparison classes (left/right and body parts merged)."""
    lut = np.full(256, 255, np.uint8)
    for i, n in enumerate(model_names):
        n = n.lower().replace("_", "-")
        n = "skin" if n in ("face", "arms", "legs", "left-arm", "right-arm", "left-leg", "right-leg") else n
        n = "shoes" if n in ("left-shoe", "right-shoe") else n
        assert n in EVAL, f"no comparison class for model class {n!r}"
        lut[i] = EVAL.index(n)
    return lut


# 1. Our final model.
ours = Segmenter(OUR_CHECKPOINT, flip_tta=True)
ours_lut = eval_lut(ours.names)

# 2. The same network before our training: ImageNet encoder weights, decoder not trained yet.
torch.manual_seed(0)
start = build_model(ours.cfg.model, ours.num_classes, pretrained=True)
torch.save({"model": start.state_dict(), "config": ours.cfg.to_dict(), "epoch": 0}, BEFORE_PATH)
before = Segmenter(BEFORE_PATH, flip_tta=True)
before_lut = eval_lut(before.names)
del start

# 3. A public clothes model from Hugging Face, also trained on ATR by someone else.
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
public_proc = AutoImageProcessor.from_pretrained(PUBLIC_MODEL)
public = SegformerForSemanticSegmentation.from_pretrained(PUBLIC_MODEL).to(ours.device).eval()
public_lut = eval_lut([public.config.id2label[i] for i in range(public.config.num_labels)])


@torch.no_grad()
def predict_public(image):
    inputs = public_proc(images=Image.fromarray(image), return_tensors="pt").to(ours.device)
    logits = public(**inputs).logits
    logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].to(torch.uint8).cpu().numpy()


MODELS = {
    "before training (ImageNet start)": lambda im: before_lut[before.predict(im)],
    "public model": lambda im: public_lut[predict_public(im)],
    "our final model": lambda im: ours_lut[ours.predict(im)],
}
print("models ready:", ", ".join(MODELS), f"| public model: {PUBLIC_MODEL}")'''),
    md("## 8. Score the three models on the same photos"),
    code('''K = len(EVAL)
garment_idx = [EVAL.index(c) for c in GARMENTS]


def scores(cm):
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    g = np.zeros((2, 2))
    is_g = np.isin(np.arange(K), garment_idx)
    for a in (0, 1):
        for b in (0, 1):
            g[a, b] = cm[np.ix_(is_g == a, is_g == b)].sum()
    return {"garment mIoU": np.nanmean(iou[garment_idx]), "mIoU (13 classes)": np.nanmean(iou),
            "clothing vs rest IoU": g[1, 1] / max(1, g[1, 1] + g[0, 1] + g[1, 0]),
            "pixel accuracy": inter.sum() / max(1, cm.sum()), **{c: iou[k] for k, c in enumerate(EVAL) if k}}


results, cms, preds = {}, {}, {name: {} for name in MODELS}
show_ids = {clean[i]["id"] for i in np.random.default_rng(0).permutation(len(clean))[:6]}
for name, fn in MODELS.items():
    cm = np.zeros((K, K), np.int64)
    t0 = time.time()
    for s in clean:
        image = load_image(s["photo"])
        p = fn(image)
        valid = s["gt"] != 255
        cm += np.bincount(s["gt"][valid].astype(np.int64) * K + p[valid], minlength=K * K).reshape(K, K)
        if s["id"] in show_ids:
            preds[name][s["id"]] = p
    results[name] = {**scores(cm), "ms per photo": 1000 * (time.time() - t0) / len(clean)}
    cms[name] = cm
    print(f"{name}: garment mIoU {results[name]['garment mIoU']:.4f}", flush=True)

table = pd.DataFrame(results)
print()
print(f"CCP, {len(clean)} photos not in ATR, original size")
print(table.round(4).to_string())
os.makedirs(f"{DRIVE_ROOT}/reports", exist_ok=True)
table.to_csv(f"{DRIVE_ROOT}/reports/ccp_comparison.csv")
print("saved", f"{DRIVE_ROOT}/reports/ccp_comparison.csv")'''),
    md("## 9. Six example photos side by side"),
    code('''import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAL = np.array([[0, 0, 0], [128, 0, 0], [255, 0, 0], [170, 0, 51], [255, 85, 0], [0, 128, 0], [0, 85, 85], [0, 0, 85],
                [85, 51, 0], [255, 255, 0], [85, 255, 170], [0, 119, 221], [52, 86, 128]], np.uint8)
PAL_IGN = np.vstack([PAL, np.full((256 - len(PAL), 3), 255, np.uint8)])   # ignored pixels shown white
ids = sorted(preds["our final model"])
cols = ["photo", "CCP labels"] + list(MODELS)
fig, axes = plt.subplots(len(ids), len(cols), figsize=(3 * len(cols), 4.4 * len(ids)), squeeze=False)
for r, sid in enumerate(ids):
    s = next(x for x in clean if x["id"] == sid)
    pics = [load_image(s["photo"]), PAL_IGN[s["gt"]]] + [PAL[preds[m][sid]] for m in MODELS]
    for c, (title, pic) in enumerate(zip(cols, pics)):
        axes[r, c].imshow(pic)
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(title, fontsize=11)
handles = [mpatches.Patch(color=PAL[k] / 255, label=n) for k, n in enumerate(EVAL)] + [mpatches.Patch(color="white", ec="gray", label="ignored")]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=10, frameon=False)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DRIVE_ROOT}/reports/ccp_examples.png", dpi=110, bbox_inches="tight")
plt.show()
print("saved", f"{DRIVE_ROOT}/reports/ccp_examples.png")'''),
    md("""## What to send back

Everything cells 6 and 8 print, and the picture from cell 9."""),
]
ccp_nb = nbf.v4.new_notebook(cells=ccp_cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.validate(ccp_nb)
nbf.write(ccp_nb, "notebooks/colab_ccp_compare.ipynb")

# ---- Which model is better? YOLO26x-sem trained 12 epochs on ATR, against our SegFormer-B3 and two public parsers. ----
YOLO_VERSION = "8.4.160"   # first release with semantic models: 8.4.52 (2026-05-20); pinned to the version tested here
COMPARE_INSTALL = '''!pip install -q "ultralytics==@YOLO_VERSION@"
import ultralytics
print("ultralytics", ultralytics.__version__)
try:
    from ultralytics.engine.results import SemanticMask   # exists only in versions with semantic segmentation
except ImportError:
    raise AssertionError("This ultralytics version has no semantic segmentation; it needs 8.4.52 or newer.")
print("semantic segmentation available")'''

COMPARE_TRAIN = '''import os
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_gb >= 20, (f"This GPU has {gpu_gb:.0f} GB. Training YOLO26x-sem needs the L4 (about 22 GB). "
                      "Choose Runtime > Change runtime type > L4 GPU, then run cells 1 to 5 and this cell again.")
os.chdir(CODE_DIR)
results_csv = f"{DRIVE_ROOT}/runs_yolo/{YOLO_RUN}/results.csv"


def epochs_done():
    return max(0, sum(1 for _ in open(results_csv)) - 1) if os.path.exists(results_csv) else 0


# Up to 3 attempts: a crash inside the GPU library (for example "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")
# is often a one-off. Each attempt continues from the last finished epoch; a crash inside an epoch
# repeats only that epoch. A failed "!python" command does not stop the notebook, hence the checks.
for attempt in range(1, 4):
    !python scripts/train_yolo_sem.py --drive-root "{DRIVE_ROOT}" --model {YOLO_MODEL} --run-name {YOLO_RUN} --epochs {EPOCHS} --imgsz {IMGSZ} --batch {BATCH} --height {HEIGHT} --width {WIDTH}
    if epochs_done() >= EPOCHS and os.path.exists(YOLO_BEST):
        break
    print(f"*** attempt {attempt} of 3 stopped with {epochs_done()} of {EPOCHS} epochs finished ***", flush=True)

done = epochs_done()
assert done >= EPOCHS and os.path.exists(YOLO_BEST), (
    f"Training stopped after {done} of {EPOCHS} epochs, three times. Send me the last 30 lines of the output above.")
print(f"YOLO training finished: {done} epochs, weights at {YOLO_BEST}")'''

COMPARE_ATR = '''import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, f"{CODE_DIR}/src")
from clothseg.data.splits import load_splits
from clothseg.external import YoloSemSegmenter, score_atr
from clothseg.inference import Segmenter

val_ids = load_splits(ATR_CACHE)["val"]
assert os.path.exists(YOLO_BEST), f"No YOLO weights at {YOLO_BEST}. Run cell 6 first."
CANDIDATES = {
    "YOLO26x-sem, 12 epochs": lambda: YoloSemSegmenter(YOLO_BEST, imgsz=IMGSZ),
    "SegFormer-B3, 12 epochs (same budget)": lambda: Segmenter(f"{DRIVE_ROOT}/runs/r1_segformer_b3/checkpoints/best.pt"),
    "SegFormer-B3, 30 epochs (our final, reference)": lambda: Segmenter(f"{DRIVE_ROOT}/runs/final_segformer_b3_e30/checkpoints/best.pt"),
}
CORE = ["upper-clothes", "pants", "skirt", "dress"]
atr_rows = {}
for name, make in CANDIDATES.items():
    print("scoring", name, "on", len(val_ids), "ATR validation photos ...", flush=True)
    model = make()
    t0 = time.time()
    m = score_atr(model.predict, ATR_CACHE, val_ids)
    atr_rows[name] = {"garment mIoU": m["garment_miou"], "mIoU (15)": m["miou"], "core garments": m["core_garment_miou"],
                      **{c: m["iou"][c] for c in CORE + ["belt", "scarf", "sunglasses", "shoes", "bag", "hat"]},
                      "ms per photo": 1000 * (time.time() - t0) / len(val_ids)}
    del model
    torch.cuda.empty_cache()
atr_table = pd.DataFrame(atr_rows)
print()
print("ATR VALIDATION, 690 photos, original size, no flip averaging")
print(atr_table.round(4).to_string())
atr_table.to_csv(f"{DRIVE_ROOT}/reports/compare_yolo_atr_val.csv")
y, b = atr_table["YOLO26x-sem, 12 epochs"], atr_table["SegFormer-B3, 12 epochs (same budget)"]
gain = y["garment mIoU"] - b["garment mIoU"]
worst = min(y[c] - b[c] for c in CORE)
ok = gain >= 0.005 and worst >= -0.01
print()
print(f"YOLO26x-sem against SegFormer-B3, both 12 epochs: garment mIoU {100 * gain:+.2f} points; largest drop on a core garment {100 * worst:+.2f} points -> "
      + ("YOLO26x-sem IS BETTER by the rule" if ok else "YOLO26x-sem is NOT better by the rule") + " (at least +0.50, no core garment below -1.00)")'''

COMPARE_CCP = '''import time
import torch
from clothseg.external import FashnParser, SchpAtr, YoloSemSegmenter
from clothseg.inference import Segmenter
from clothseg.labels import class_names

assert os.path.exists(YOLO_BEST), f"No YOLO weights at {YOLO_BEST}. Run cell 6 first."
K = len(EVAL)
NAMES15 = class_names(15)


def lut_from(names, rule):
    lut = np.full(256, 255, np.uint8)
    for i, n in enumerate(names):
        lut[i] = EVAL.index(rule(n))
    return lut


def rule15(n):
    return "skin" if n in ("face", "arms", "legs") else n


# FASHN's own classes: its body parts become skin; it has no shoes class ("feet" is a body class),
# so shoes are left out of its comparison; jewelry has no comparison class and counts as background.
FASHN_RULE = {"background": "background", "face": "skin", "hair": "hair", "top": "upper-clothes", "dress": "dress",
              "skirt": "skirt", "pants": "pants", "belt": "belt", "bag": "bag", "hat": "hat", "scarf": "scarf",
              "glasses": "sunglasses", "arms": "skin", "hands": "skin", "legs": "skin", "feet": "skin", "torso": "skin",
              "jewelry": "background"}

MODELS = {
    "our final (B3, 30 ep)": (lambda: Segmenter(f"{DRIVE_ROOT}/runs/final_segformer_b3_e30/checkpoints/best.pt"), lambda m: lut_from(NAMES15, rule15)),
    "our B3, 12 ep": (lambda: Segmenter(f"{DRIVE_ROOT}/runs/r1_segformer_b3/checkpoints/best.pt"), lambda m: lut_from(NAMES15, rule15)),
    "YOLO26x-sem, 12 ep": (lambda: YoloSemSegmenter(YOLO_BEST, imgsz=IMGSZ), lambda m: lut_from(NAMES15, rule15)),
    "SCHP-ATR": (lambda: SchpAtr(), lambda m: lut_from(NAMES15, rule15)),
    "FASHN parser": (lambda: FashnParser(), lambda m: lut_from(m.names, FASHN_RULE.__getitem__)),
}
SHOES = EVAL.index("shoes")
G10 = [EVAL.index(c) for c in GARMENTS]
G9 = [k for k in G10 if k != SHOES]


def ious(cm):
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


from clothseg.diagnose import Diagnosis

ccp_rows, ccp_preds = {}, {name: {} for name in MODELS}
diags = {name: Diagnosis(EVAL, GARMENTS) for name in MODELS}   # where each model's errors are (cell 13)
show_ids = {clean[i]["id"] for i in np.random.default_rng(0).permutation(len(clean))[:6]}
for name, (make, make_lut) in MODELS.items():
    print("scoring", name, "on", len(clean), "CCP photos ...", flush=True)
    model = make()
    lut = make_lut(model)
    cm10, cm9 = np.zeros((K, K), np.int64), np.zeros((K, K), np.int64)
    t0 = time.time()
    for s in clean:
        image = load_image(s["photo"])
        p = lut[model.predict(image)]
        gt = s["gt"]
        v = gt != 255
        cm10 += np.bincount(gt[v].astype(np.int64) * K + p[v], minlength=K * K).reshape(K, K)
        diags[name].update(gt, p, s["id"])
        v9 = v & (gt != SHOES)     # shoes left out for everyone in the 9-garment score
        cm9 += np.bincount(gt[v9].astype(np.int64) * K + p[v9], minlength=K * K).reshape(K, K)
        if s["id"] in show_ids:
            ccp_preds[name][s["id"]] = p
    i10, i9 = ious(cm10), ious(cm9)
    ccp_rows[name] = {"garment mIoU (10)": np.nanmean(i10[G10]) if name != "FASHN parser" else np.nan,
                      "garment mIoU without shoes (9)": np.nanmean(i9[G9]),
                      "mIoU (13)": np.nanmean(i10) if name != "FASHN parser" else np.nan,
                      **{EVAL[k]: i10[k] for k in range(1, K)},
                      "ms per photo": 1000 * (time.time() - t0) / len(clean)}
    if name == "FASHN parser":
        ccp_rows[name]["shoes"] = np.nan
    del model
    torch.cuda.empty_cache()

ccp_table = pd.DataFrame(ccp_rows)
print()
print(f"CCP, {len(clean)} photos none of the models was trained on, original size, no flip averaging")
print("(FASHN has no shoes class, so its 10-garment and 13-class scores are not comparable: use the 9-garment score)")
print(ccp_table.round(4).to_string())
ccp_table.to_csv(f"{DRIVE_ROOT}/reports/compare_models_ccp.csv")
ours = ccp_table["our final (B3, 30 ep)"]
print()
for name in ccp_table.columns[1:]:
    other = ccp_table[name]
    key = "garment mIoU without shoes (9)"
    gain = other[key] - ours[key]
    worst = min(other[c] - ours[c] for c in ("upper-clothes", "pants", "skirt", "dress"))
    verdict = "BETTER than ours by the rule" if gain >= 0.005 and worst >= -0.01 else "not better than ours by the rule"
    print(f"{name}: {key} {100 * gain:+.2f} points against our final model; largest core-garment drop {100 * worst:+.2f} -> {verdict}")'''

COMPARE_FIGURE = '''import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAL = np.array([[0, 0, 0], [128, 0, 0], [255, 0, 0], [170, 0, 51], [255, 85, 0], [0, 128, 0], [0, 85, 85], [0, 0, 85],
                [85, 51, 0], [255, 255, 0], [85, 255, 170], [0, 119, 221], [52, 86, 128]], np.uint8)
PAL_IGN = np.vstack([PAL, np.full((256 - len(PAL), 3), 255, np.uint8)])
ids = sorted(ccp_preds["our final (B3, 30 ep)"])
cols = ["photo", "CCP labels"] + list(MODELS)
fig, axes = plt.subplots(len(ids), len(cols), figsize=(2.6 * len(cols), 4.0 * len(ids)), squeeze=False)
for r, sid in enumerate(ids):
    s = next(x for x in clean if x["id"] == sid)
    pics = [load_image(s["photo"]), PAL_IGN[s["gt"]]] + [PAL[ccp_preds[m][sid]] for m in MODELS]
    for c, (title, pic) in enumerate(zip(cols, pics)):
        axes[r, c].imshow(pic)
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(title, fontsize=10)
handles = [mpatches.Patch(color=PAL[k] / 255, label=n) for k, n in enumerate(EVAL)] + [mpatches.Patch(color="white", ec="gray", label="ignored")]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DRIVE_ROOT}/reports/compare_models_ccp_examples.png", dpi=110, bbox_inches="tight")
plt.show()
print("saved", f"{DRIVE_ROOT}/reports/compare_models_ccp_examples.png")'''

assert "clothing-co-parsing.git" in ccp_cells[10]["source"] and "CCP_TO_EVAL" in ccp_cells[12]["source"]
WHY_ALL = 'import pickle, re, sys\nimport matplotlib.pyplot as plt\nfrom PIL import Image\nsys.path.insert(0, f"{CODE_DIR}/src")\nfrom clothseg.data.splits import load_splits\nfrom clothseg.diagnose import Diagnosis, explain\nfrom clothseg.external import score_atr\nfrom clothseg.labels import CLASSES_15, GARMENT_NAMES, build_lut, palette\n\npd.set_option("display.width", 250)\npickle.dump(diags, open(f"{DRIVE_ROOT}/reports/why_ccp.pkl", "wb"))\nOURS = "our final (B3, 30 ep)"\nPAL = np.array([[0, 0, 0], [128, 0, 0], [255, 0, 0], [170, 0, 51], [255, 85, 0], [0, 128, 0], [0, 85, 85], [0, 0, 85],\n                [85, 51, 0], [255, 255, 0], [85, 255, 170], [0, 119, 221], [52, 86, 128]], np.uint8)\nPAL_IGN = np.vstack([PAL, np.full((256 - len(PAL), 3), 255, np.uint8)])\nby_id = {s["id"]: s for s in clean}\nours_model = MODELS[OURS][0]()\nours_lut = MODELS[OURS][1](ours_model)\n\n\ndef show(rows, title, out):\n    """rows: list of lists of (title, picture); one figure, saved to Drive."""\n    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(2.5 * len(rows[0]), 4.2 * len(rows)), squeeze=False)\n    for r, pics in enumerate(rows):\n        for c, (t, pic) in enumerate(pics):\n            axes[r, c].imshow(pic)\n            axes[r, c].axis("off")\n            axes[r, c].set_title(t, fontsize=9)\n    fig.suptitle(title, fontsize=11)\n    fig.tight_layout()\n    fig.savefig(out, dpi=90, bbox_inches="tight")\n    plt.show()\n\n\ndef yolo_on_atr():\n    """YOLO and SegFormer-B3 trained on the same ATR photos for 12 epochs: the cleanest test of the architecture."""\n    val_ids = load_splits(ATR_CACHE)["val"]\n    g15 = [n for n in CLASSES_15 if n in GARMENT_NAMES]\n    makers = {"YOLO26x-sem": MODELS["YOLO26x-sem, 12 ep"][0], "SegFormer-B3": MODELS["our B3, 12 ep"][0]}\n    d = {}\n    for n, make in makers.items():\n        print(f"   diagnosing {n} on {len(val_ids)} ATR validation photos ...", flush=True)\n        m = make()\n        d[n] = Diagnosis(CLASSES_15, g15)\n        score_atr(m.predict, ATR_CACHE, val_ids, diagnosis=d[n])\n        del m\n        torch.cuda.empty_cache()\n    pickle.dump(d, open(f"{DRIVE_ROOT}/reports/why_atr_val.pkl", "wb"))\n    print("   On ATR validation, both trained on the same photos for 12 epochs:")\n    for line in explain(d["YOLO26x-sem"], d["SegFormer-B3"], "YOLO", "B3", top=6):\n        print("    -", line)\n    models = {n: make() for n, make in makers.items()}\n    pal15, lut15, rows = palette(15), build_lut(15), []\n    for gap, k in d["YOLO26x-sem"].worst_against(d["SegFormer-B3"], 6):\n        gt = lut15[np.asarray(Image.open(f"{ATR_CACHE}/masks_orig/{int(k):06d}.png"))]\n        image = np.asarray(Image.open(f"{ATR_CACHE}/images/{int(k):06d}.jpg").convert("RGB").resize((gt.shape[1], gt.shape[0])))\n        rows.append([("ATR photo", image), ("labels", np.where(gt[..., None] == 255, 255, pal15[np.minimum(gt, 14)]).astype(np.uint8))]\n                    + [(f"{n} ({d[n].image_score(k):.2f})", pal15[m.predict(image)]) for n, m in models.items()])\n    show(rows, "YOLO26x-sem against SegFormer-B3 on ATR: the 6 photos where YOLO falls furthest behind",\n         f"{DRIVE_ROOT}/reports/why_yolo_worst_atr.png")\n    del models\n    torch.cuda.empty_cache()\n\n\n# ---- One section per model: why it scores lower than our final model, in numbers and in photos.\nfor name, (make, make_lut) in MODELS.items():\n    if name == OURS:\n        continue\n    print()\n    print("=" * 110)\n    print(f"WHY {name.upper()} IS BELOW OUR FINAL MODEL ({len(clean)} CCP photos, none seen in training)")\n    for line in explain(diags[name], diags[OURS], name, "ours", top=5):\n        print(" -", line)\n    exclude = ("shoes",) if name == "FASHN parser" else ()   # FASHN has no shoes class: not counted when choosing photos\n    if exclude:\n        print(" - FASHN has no shoes class (its \'feet\' is a body class), so its shoe pixels count as skin: a different label definition, not a mistake.")\n    model = make()\n    lut = make_lut(model)\n    rows = []\n    for gap, k in diags[name].worst_against(diags[OURS], 6, exclude):\n        image = load_image(by_id[k]["photo"])\n        rows.append([(f"photo {k}", image), ("CCP labels", PAL_IGN[by_id[k]["gt"]]),\n                     (f"ours ({diags[OURS].image_score(k, exclude):.2f})", PAL[ours_lut[ours_model.predict(image)]]),\n                     (f"{name} ({diags[name].image_score(k, exclude):.2f})", PAL[lut[model.predict(image)]])])\n    del model\n    torch.cuda.empty_cache()\n    show(rows, f"{name}: the 6 CCP photos where it falls furthest behind our final model (garment IoU per photo)",\n         f"{DRIVE_ROOT}/reports/why_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + "_worst_ccp.png")\n    if name.startswith("YOLO"):\n        yolo_on_atr()\ndel ours_model\ntorch.cuda.empty_cache()\n\n# ---- All models side by side: which kind of error each one makes.\nprint()\nprint("=" * 110)\nprint("ALL MODELS SIDE BY SIDE (CCP)")\nprint()\nprint("Outlines, insides and garment size: pixel accuracy in % and garment IoU by size")\nrows = {n: {**{k: 100 * v for k, v in d.outline_rows().items()}, **d.size_rows()} for n, d in diags.items()}\nprint(pd.DataFrame(rows).round(3).to_string())\nprint()\nprint("Missed garments: % of photos with the garment where less than 20% of it was found")\nprint(pd.DataFrame({n: {r["garment"]: r["missed (%)"] for r in d.presence_rows()} for n, d in diags.items()}).round(1).to_string())\nprint()\nprint("Invented garments: % of photos without the garment where the model shows at least 0.2% of the photo as it")\nprint(pd.DataFrame({n: {r["garment"]: r["invented (% of photos without it)"] for r in d.presence_rows()} for n, d in diags.items()}).round(1).to_string())'
compare_cells = [
    md("""# Clothes segmentation: which model is better?

**What it does:** trains **YOLO26x-sem** (Ultralytics, 40.2 M parameters, semantic segmentation, released May 2026) for **12 epochs on ATR**, the same data, classes and epoch budget as our SegFormer-B3 12-epoch run, and then compares it with our models and with two public human parsers:

| model | trained by | on |
|---|---|---|
| our final model, SegFormer-B3, 30 epochs | us | ATR |
| our SegFormer-B3, 12 epochs (`r1_segformer_b3`) | us | ATR |
| YOLO26x-sem, 12 epochs | this notebook | ATR |
| SCHP-ATR (`pirocheto/schp-atr-18`) | others | ATR |
| FASHN Human Parser (`fashn-ai/fashn-human-parser`, SegFormer-B4) | others | their own data |

**Two tests:**

1. **ATR validation** (690 photos, original size, no flip averaging): YOLO26x-sem against SegFormer-B3, both 12 epochs. Same budget, so this is the fair architecture test.
2. **CCP** (986 photos none of the models was trained on): all five. FASHN has no shoes class ("feet" is a body class), so every model is also scored without shoes, and that 9-garment score is the one used for FASHN.

**Rule, fixed before the result** (PLAN.md, section 48): a model is better only if its garment mIoU is at least **+0.50 points** higher and **no core garment** (upper-clothes, pants, skirt, dress) is more than **1.00 point** lower.

Use the **L4 GPU**. Run the cells from top to bottom; cell 3 asks for `clothseg_code.zip` once. Training takes a few hours (not measured yet; the first epoch line shows the speed). If the session ends, run cells 1 to 5 and cell 6 again: it continues from the last finished epoch saved on Drive. Once training has finished, cells 7 to 11 can be run in any later session after cells 1 to 5."""),
    md("## 1. Settings"),
    code('''REPO_URL = ""

DRIVE_ROOT = "/content/drive/MyDrive/clothseg"
CODE_DIR = "/content/clothes-segmentation"
CODE_VERSION = "@VERSION@"
HEIGHT, WIDTH = 576, 384
SESSION_BUDGET_MIN = 600

YOLO_MODEL = "yolo26x-sem.pt"   # downloaded by Ultralytics; pretrained on Cityscapes
YOLO_RUN = "yolo26x_sem_e12"     # folder inside <DRIVE_ROOT>/runs_yolo
EPOCHS = 12
IMGSZ = 576                      # long side of our 576x384 photos
BATCH = 16                       # if cell 6 stops with "CUDA out of memory", set 8 and run cells 1 and 6 again
YOLO_BEST = f"{DRIVE_ROOT}/runs_yolo/{YOLO_RUN}/weights/best.pt"
ATR_CACHE = f"/content/data/atr_{HEIGHT}x{WIDTH}"'''.replace("@VERSION@", VERSION)),
    md("## 2. Mount Drive and show the environment"),
    code(train_cell("drive.mount")),
    md("## 3. Get the code"),
    code(train_cell("def code_version")),
    md("## 4. Install requirements and prepare ATR"),
    code(train_cell("pip install") + '\n!python scripts/prepare_data.py --drive-root "{DRIVE_ROOT}" --height {HEIGHT} --width {WIDTH}'),
    md("## 5. Install Ultralytics"),
    code(COMPARE_INSTALL.replace("@YOLO_VERSION@", YOLO_VERSION)),
    md("""## 6. Train YOLO26x-sem for 12 epochs on ATR

Same training and validation photos as our runs, the same 15 classes (left and right merged), unlabelled pixels ignored. Ultralytics' own training recipe (its loss, optimiser and augmentation), because that is how the model is meant to be trained. Weights and `results.csv` are saved to `<DRIVE_ROOT>/runs_yolo/yolo26x_sem_e12` after every epoch."""),
    code(COMPARE_TRAIN),
    md("""## 7. ATR validation: YOLO26x-sem against SegFormer-B3, both 12 epochs

Scored exactly like `scripts/evaluate.py`: each photo at its original size, no flip averaging. The final 30-epoch model is shown for reference. About 10 minutes."""),
    code(COMPARE_ATR),
    md("## 8. Download CCP"),
    code(ccp_cells[10]["source"]),
    md("## 9. Read the CCP labels, map them, and remove photos that are also in ATR"),
    code(ccp_cells[12]["source"]),
    md("""## 10. Score all five models on CCP

Every model sees each photo at its original size, without flip averaging. Face, arms and legs (and FASHN's hands, feet and torso) become "skin"; FASHN's "top" is upper-clothes, its "glasses" sunglasses and its "jewelry" background. About 15 minutes."""),
    code(COMPARE_CCP),
    md("## 11. Six example photos side by side"),
    code(COMPARE_FIGURE),
    md("""## What to send back

The executed notebook, or everything cells 6 (first and last epoch lines), 7 and 10 print, and the picture from cell 11. The reasons are in `colab_why_ours_wins.ipynb`. The tables are also saved on Drive in `clothseg/reports/`: `compare_yolo_atr_val.csv`, `compare_models_ccp.csv` and `compare_models_ccp_examples.png`."""),
]
compare_nb = nbf.v4.new_notebook(cells=compare_cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.validate(compare_nb)
nbf.write(compare_nb, "notebooks/colab_model_compare.ipynb")

# ---- Why our model beats every other model: scoring and all explanations, nothing else. ----
why_cells = [
    md("""# Clothes segmentation: why our model beats every other model

Scores five models on the same 986 CCP photos that none of them was trained on, then explains, **in one cell**, why each of the others is below our final model:

- the garments where it loses most, and which class those pixels went to;
- the kind of error: outlines, small garments, garments missed or invented;
- the six photos where it falls furthest behind ours.

For YOLO26x-sem the explanation also covers ATR validation, where it was trained on the same photos as our SegFormer-B3. That is the cleanest test of the architecture.

| model | trained by | on |
|---|---|---|
| our final model, SegFormer-B3, 30 epochs | us | ATR |
| our SegFormer-B3, 12 epochs | us | ATR |
| YOLO26x-sem, 12 epochs (trained with `colab_model_compare.ipynb`) | us | ATR |
| SCHP-ATR | others | ATR |
| FASHN Human Parser (SegFormer-B4) | others | their own data |

A **T4 GPU is enough**. About 40 minutes. Run the cells from top to bottom; cell 3 asks for `clothseg_code.zip` only if Drive has an older code version."""),
    md("## 1. Settings"),
    code('''REPO_URL = ""

DRIVE_ROOT = "/content/drive/MyDrive/clothseg"
CODE_DIR = "/content/clothes-segmentation"
CODE_VERSION = "@VERSION@"
HEIGHT, WIDTH = 576, 384
SESSION_BUDGET_MIN = 600   # used by a helper in cell 2; nothing is trained here

IMGSZ = 576
YOLO_BEST = f"{DRIVE_ROOT}/runs_yolo/yolo26x_sem_e12/weights/best.pt"
ATR_CACHE = f"/content/data/atr_{HEIGHT}x{WIDTH}"'''.replace("@VERSION@", VERSION)),
    md("## 2. Mount Drive and show the environment"),
    code(train_cell("drive.mount")),
    md("## 3. Get the code"),
    code(train_cell("def code_version")),
    md("## 4. Install requirements and Ultralytics, prepare ATR"),
    code(train_cell("pip install") + '\n!python scripts/prepare_data.py --drive-root "{DRIVE_ROOT}" --height {HEIGHT} --width {WIDTH}\n'
         + COMPARE_INSTALL.replace("@YOLO_VERSION@", YOLO_VERSION)),
    md("## 5. Download CCP, read its labels, remove photos that are also in ATR"),
    code(ccp_cells[10]["source"] + "\n\n" + ccp_cells[12]["source"]),
    md("## 6. Score all five models on CCP"),
    code(COMPARE_CCP.replace("Run cell 6 first.", "Train it first with notebooks/colab_model_compare.ipynb.")),
    md("## 7. Why our model beats every other model"),
    code(WHY_ALL),
    md("""## What to send back

Everything cells 6 and 7 print, and the pictures from cell 7. They are also saved on Drive in `clothseg/reports/` (`compare_models_ccp.csv`, `why_*.png`)."""),
]
why_nb = nbf.v4.new_notebook(cells=why_cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}})
nbf.validate(why_nb)
nbf.write(why_nb, "notebooks/colab_why_ours_wins.ipynb")

files_in_zip = ["VERSION", "requirements.txt"] + sorted(
    str(p).replace("\\", "/") for d in ("src", "scripts") for p in Path(d).rglob("*.py"))
with zipfile.ZipFile("clothseg_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files_in_zip:
        z.write(f, f)
print(f"version {VERSION}: notebooks/colab_eval.ipynb written, colab_train.ipynb updated, zip has {len(files_in_zip)} files")
