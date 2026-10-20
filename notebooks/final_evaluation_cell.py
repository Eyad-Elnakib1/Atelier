# Final evaluation of the final model. Paste this whole block into ONE new cell of
# notebooks/colab_eval.ipynb, after cells 1 to 4 have run, and run it once.
#
# What it does, in this order:
#   1. validation split, without and with flip averaging;
#   2. a safety check on the validation result;
#   3. decides on flip averaging by the rule fixed in PLAN.md, section 17.11;
#   4. test split, once, in both settings; prints and shows the results.

import json, os, subprocess, sys
import pandas as pd
from IPython.display import Image, display

RUN_NAME = "final_segformer_b3_e30"
CHECKPOINT = f"{DRIVE_ROOT}/runs/{RUN_NAME}/checkpoints/best.pt"
EXPECTED_VAL_MIOU = 0.7866      # validation mIoU of this checkpoint reported during training
FLIP_MIN_GAIN = 0.002           # flip averaging is adopted if it adds at least 0.2 points of mIoU

os.chdir(CODE_DIR)
assert os.path.exists(CHECKPOINT), f"No checkpoint at {CHECKPOINT}."
assert "--flip-tta" in open("scripts/evaluate.py").read(), (
    "This session has an older version of the code. Open the new colab_eval.ipynb from the project folder, "
    "run cells 1 to 4 (upload the new clothseg_code.zip when asked), then run this cell again.")


def evaluate(split, flip):
    """Run scripts/evaluate.py and return (metrics, output folder)."""
    out = f"{DRIVE_ROOT}/reports/eval_{RUN_NAME}_{split}" + ("_flip" if flip else "")
    cmd = [sys.executable, "scripts/evaluate.py", "--checkpoint", CHECKPOINT, "--split", split, "--out-dir", out]
    print(f"evaluating {split}{' with flip averaging' if flip else ''} ...", flush=True)
    r = subprocess.run(cmd + (["--flip-tta"] if flip else []), capture_output=True, text=True)
    assert r.returncode == 0, "The evaluation failed. Send me this message:\n" + (r.stdout + r.stderr)[-3000:]
    return json.load(open(f"{out}/metrics.json")), out


KEYS = ("miou", "garment_miou", "core_garment_miou", "clothing_binary_iou", "pixel_acc", "mean_class_acc",
        "mean_dice", "boundary_miou", "garment_boundary_miou", "ms_per_image")


def table(title, plain, flip):
    print(f"\n{title}  ({plain['num_images']} images)")
    print(f"{'metric':<24}{'plain':>9}{'with flip':>11}{'change':>9}")
    for k in KEYS:
        print(f"{k:<24}{plain[k]:>9.4f}{flip[k]:>11.4f}{flip[k] - plain[k]:>+9.4f}")


# 1. Validation.
val_plain, _ = evaluate("val", False)
val_flip, _ = evaluate("val", True)
table("VALIDATION", val_plain, val_flip)

# 2. Safety check before the test set is touched.
assert abs(val_plain["miou"] - EXPECTED_VAL_MIOU) < 0.01, (
    f"Validation mIoU is {val_plain['miou']:.4f}, expected about {EXPECTED_VAL_MIOU}. "
    "Something is wrong. The test set was NOT used. Send me the output above.")

# 3. Decision on flip averaging, from validation only.
gain = val_flip["miou"] - val_plain["miou"]
use_flip = gain >= FLIP_MIN_GAIN
print(f"\nflip averaging changes validation mIoU by {100 * gain:+.2f} points -> "
      + ("ADOPTED (rule: at least +0.20)" if use_flip else "NOT adopted (rule: at least +0.20)"))

# 4. Test split, once.
test_plain, out_plain = evaluate("test", False)
test_flip, out_flip = evaluate("test", True)
table("TEST", test_plain, test_flip)
head, head_dir = (test_flip, out_flip) if use_flip else (test_plain, out_plain)
print(f"\nHEADLINE TEST RESULT ({'with' if use_flip else 'without'} flip averaging): "
      f"mIoU {head['miou']:.4f} | garment mIoU {head['garment_miou']:.4f} | clothing binary IoU {head['clothing_binary_iou']:.4f} | "
      f"pixel accuracy {head['pixel_acc']:.4f} | boundary mIoU {head['boundary_miou']:.4f}")

print("\nPer-class results of the headline test evaluation")
print(pd.read_csv(f"{head_dir}/per_class.csv").round(4).to_string(index=False))
print(f"\nfiles saved in {head_dir}")
for name in ("confusion_matrix.png", "worst_examples.png", "best_examples.png"):
    display(Image(f"{head_dir}/{name}"))
