"""Prove that resuming works before any long run is started.

Trains the same tiny experiment twice:
  A) without interruption;
  B) stopped in the middle of an epoch, then resumed twice.
Then compares the two logs step by step.

PASS means: B continued at the right step, saw exactly the same batches as A in the
same order, and its losses match A closely (GPU arithmetic is not bit-reproducible,
so tiny differences are normal).

Example (Colab):
    python scripts/resume_test.py --drive-root /content/drive/MyDrive/clothseg
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_log(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--local-root", type=Path, default=Path("/content/data"))
    p.add_argument("--model", default="segformer_b2")
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--loss-tolerance", type=float, default=0.1, help="Largest accepted loss difference between the two runs.")
    a = p.parse_args()

    steps_per_epoch, epochs = 12, 2  # 24 steps in total
    common = [
        sys.executable, str(HERE / "train.py"), "--drive-root", str(a.drive_root), "--local-root", str(a.local_root),
        "--model", a.model, "--height", str(a.height), "--width", str(a.width), "--batch-size", str(a.batch_size),
        "--epochs", str(epochs), "--limit-train", str(steps_per_epoch * a.batch_size), "--limit-val", "32",
        "--warmup-steps", "5", "--log-every", "1", "--workers", str(a.workers),
    ]
    runs = {n: a.drive_root / "runs" / n for n in ("_resume_test_A", "_resume_test_B")}
    for d in runs.values():
        shutil.rmtree(d, ignore_errors=True)

    def run(name: str, *extra: str) -> None:
        print(f"\n===== {name} {' '.join(extra)} =====", flush=True)
        r = subprocess.run(common + ["--run-name", name, *extra])
        if r.returncode != 0:
            sys.exit(f"RESUME TEST FAILED: the training command for {name} exited with an error (see above).")

    run("_resume_test_A")                                # uninterrupted
    run("_resume_test_B", "--stop-after-steps", "7")     # dies in the middle of epoch 1
    run("_resume_test_B", "--stop-after-steps", "17")    # resumes, dies in the middle of epoch 2
    run("_resume_test_B")                                # resumes and finishes

    log_a, log_b = read_log(runs["_resume_test_A"] / "train_log.csv"), read_log(runs["_resume_test_B"] / "train_log.csv")
    val_a, val_b = read_log(runs["_resume_test_A"] / "val_log.csv"), read_log(runs["_resume_test_B"] / "val_log.csv")
    problems = []
    expected = steps_per_epoch * epochs
    if [int(r["step"]) for r in log_a] != list(range(1, expected + 1)):
        problems.append(f"uninterrupted run logged steps {[r['step'] for r in log_a]}, expected 1..{expected}")
    if [r["step"] for r in log_a] != [r["step"] for r in log_b]:
        problems.append("the resumed run did not log the same steps (missing or repeated steps)")
    else:
        wrong = [r["step"] for r, s in zip(log_a, log_b) if r["batch_hash"] != s["batch_hash"]]
        if wrong:
            problems.append(f"different training batches at steps {wrong[:10]}")
        if [r["lr_encoder"] for r in log_a] != [r["lr_encoder"] for r in log_b]:
            problems.append("the learning-rate schedule differs after resuming")
        diffs = [abs(float(r["loss"]) - float(s["loss"])) for r, s in zip(log_a, log_b)]
        print(f"\nstep-by-step loss difference between the two runs: max {max(diffs):.5f}, mean {sum(diffs) / len(diffs):.5f}")
        if max(diffs) > a.loss_tolerance:
            problems.append(f"losses differ by up to {max(diffs):.4f} (tolerance {a.loss_tolerance})")
    if len(val_a) != epochs or len(val_b) != epochs:
        problems.append(f"expected {epochs} validation rows per run, got {len(val_a)} and {len(val_b)}")
    else:
        print("val mIoU  uninterrupted: " + ", ".join(r["miou"] for r in val_a) + " | resumed: " + ", ".join(r["miou"] for r in val_b))
    for name, d in runs.items():
        kept = sorted(q.name for q in (d / "checkpoints").iterdir())
        print(f"{name} checkpoints kept: {kept}")
        if "best.pt" not in kept or len([k for k in kept if k.startswith("step_")]) > 2:
            problems.append(f"{name}: expected best.pt plus at most 2 step checkpoints, found {kept}")

    for d in runs.values():
        shutil.rmtree(d, ignore_errors=True)
    print("\n" + "=" * 70)
    if problems:
        print("RESUME TEST FAILED:\n  - " + "\n  - ".join(problems))
        print("=" * 70)
        sys.exit(1)
    print("RESUME TEST PASSED: same steps, same batches, same learning rates, matching losses.")
    print("=" * 70)


if __name__ == "__main__":
    main()
