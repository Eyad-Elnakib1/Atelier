# Paste this whole block into ONE new cell at the end of the notebook you are running,
# and run it after cell 8 (the comparison) has finished.
# It adds DeepLabV3+ to the code of this session, trains it with the same settings
# as the other models, and prints the comparison again with all five models.

import os

EXTRA_MODEL = "deeplabv3plus_resnet101"
models_py = f"{CODE_DIR}/src/clothseg/models.py"
src = open(models_py).read()

if EXTRA_MODEL not in src:
    old_names = '"upernet_convnext_tiny")'
    old_branch = "        encoder = net.encoder\n    else:"
    assert src.count(old_names) == 1 and src.count(old_branch) == 1, (
        "models.py is not the version this cell expects. Do not continue; send me this message.")
    src = src.replace(old_names, '"upernet_convnext_tiny", "deeplabv3plus_resnet101")')
    src = src.replace(old_branch, (
        "        encoder = net.encoder\n"
        '    elif name == "deeplabv3plus_resnet101":\n'
        "        import segmentation_models_pytorch as smp\n"
        "\n"
        "        # The classic human-parsing baseline: ResNet-101 encoder, atrous spatial pyramid pooling decoder.\n"
        '        net = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet" if pretrained else None, classes=num_classes)\n'
        "        encoder = net.encoder\n"
        "    else:"))
    open(models_py, "w").write(src)
    print("DeepLabV3+ added to", models_py)
else:
    print("DeepLabV3+ is already in", models_py)

os.chdir(CODE_DIR)
budget = int(minutes_left())
!python scripts/train.py --drive-root "{DRIVE_ROOT}" --run-name r1_{EXTRA_MODEL} --model {EXTRA_MODEL} --height {HEIGHT} --width {WIDTH} --epochs {SCHEDULE_EPOCHS} --stop-after-epoch {ROUND1_EPOCHS} --time-budget-min {budget}

!python scripts/compare_runs.py --drive-root "{DRIVE_ROOT}" --prefix r1_ --at-epoch {ROUND1_EPOCHS} --out "{DRIVE_ROOT}/reports/compare_r1_with_extra.png"
from IPython.display import Image, display
display(Image(f"{DRIVE_ROOT}/reports/compare_r1_with_extra.png"))
