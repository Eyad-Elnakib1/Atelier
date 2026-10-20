"""Fine-tune an existing checkpoint, optionally with cyclic learning rate and weight averaging.

Two uses:

1. Weight averaging (the idea of SWA, Izmailov et al. 2018; SCHP calls it model aggregation).
   Training continues from a finished model in short cycles. In each cycle the learning
   rate starts at a fraction of the original rate and falls along a cosine curve. At the
   end of every cycle the weights are added to a running average that also contains the
   starting weights. The averaged model is the result.

2. Fine-tuning at another input size (--height/--width, usually with --no-average).

The starting checkpoint is only read, never changed. The result is written as
<drive-root>/runs/<run-name>/checkpoints/best.pt and works with scripts/evaluate.py and
scripts/segment.py. The run can be resumed at epoch boundaries: run the same command again.

Example (Colab):
    python scripts/finetune.py --drive-root /content/drive/MyDrive/clothseg \
        --init-checkpoint /content/drive/MyDrive/clothseg/runs/final_segformer_b3_e30/checkpoints/best.pt \
        --run-name swa_segformer_b3 --cycles 3 --epochs-per-cycle 3
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clothseg.checkpoint import _save  # noqa: E402
from clothseg.config import TrainConfig  # noqa: E402
from clothseg.data.dataset import AtrDataset, ResumableBatchSampler  # noqa: E402
from clothseg.data.preprocess import DONE_MARKER, cache_name  # noqa: E402
from clothseg.engine import data_fingerprint, validate  # noqa: E402
from clothseg.losses import SegLoss  # noqa: E402
from clothseg.models import build_model, load_weights  # noqa: E402

EPOCH_OFFSET = 1000  # fine-tuning epochs get their own data order and augmentations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--local-root", type=Path, default=Path("/content/data"))
    p.add_argument("--init-checkpoint", type=Path, required=True, help="Checkpoint to start from (it is not modified).")
    p.add_argument("--run-name", required=True)
    p.add_argument("--height", type=int, default=0, help="Input height. 0 = the size of the starting checkpoint.")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0, help="0 = the batch size of the starting checkpoint.")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--epochs-per-cycle", type=int, default=3)
    p.add_argument("--lr-max-scale", type=float, default=0.25, help="Learning rate at the start of a cycle, as a fraction of the original rates.")
    p.add_argument("--lr-min-scale", type=float, default=0.01, help="Learning rate at the end of a cycle, as a fraction of the original rates.")
    p.add_argument("--no-average", action="store_true", help="Keep the last weights instead of the average over cycles.")
    p.add_argument("--bn-batches", type=int, default=200, help="Training batches used to recompute BatchNorm statistics of the averaged model.")
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--time-budget-min", type=float, default=280.0, help="Stop after the current epoch once this many minutes have passed.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--limit-train", type=int, default=0, help="Use only the first N training images (tests only).")
    p.add_argument("--limit-val", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def update_bn(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool, max_batches: int) -> None:
    """Recompute BatchNorm running statistics for averaged weights (as in SWA)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    model.eval()
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the batches seen
        m.train()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            model(batch["image"].to(device, non_blocking=True))
    for m, momentum in zip(bns, saved):
        m.momentum = momentum
        m.eval()


def main() -> None:
    a = parse_args()
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not a.init_checkpoint.is_file():
        sys.exit(f"ERROR: starting checkpoint not found: {a.init_checkpoint}")
    init = torch.load(a.init_checkpoint, map_location="cpu", weights_only=False)  # written by this project
    cfg = TrainConfig.from_dict(init["config"])
    cfg.height, cfg.width = a.height or cfg.height, a.width or cfg.width
    cfg.batch_size = a.batch_size or cfg.batch_size
    cfg.epochs, cfg.pretrained = a.cycles * a.epochs_per_cycle, False
    cfg.limit_train, cfg.limit_val = a.limit_train, a.limit_val
    amp = cfg.amp and device.type == "cuda"
    average = not a.no_average

    cache_dir = a.local_root / cache_name(cfg.height, cfg.width)
    if not (cache_dir / DONE_MARKER).exists():
        sys.exit(f"ERROR: no dataset cache at {cache_dir}. Run scripts/prepare_data.py with --height {cfg.height} --width {cfg.width} first.")
    cfg.data_fingerprint = data_fingerprint(cache_dir)
    run_dir = a.drive_root / "runs" / a.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "checkpoints" / "finetune_state.pt"
    settings = {"init": str(a.init_checkpoint), "config": cfg.to_dict(), "cycles": a.cycles, "epochs_per_cycle": a.epochs_per_cycle,
                "lr_max_scale": a.lr_max_scale, "lr_min_scale": a.lr_min_scale, "average": average}

    cores = os.cpu_count() or 2
    workers = a.workers if a.workers >= 0 else (2 if cores <= 2 else min(8, cores - 2))
    train_set = AtrDataset(cache_dir, "train", cfg.num_classes, cfg.height, cfg.width, train=True, seed=cfg.seed, limit=a.limit_train or None, augment=cfg.augment)
    val_set = AtrDataset(cache_dir, "val", cfg.num_classes, cfg.height, cfg.width, train=False, limit=a.limit_val or None)
    sampler = ResumableBatchSampler(len(train_set), cfg.batch_size, cfg.seed)
    steps_per_epoch = sampler.batches_per_epoch
    if steps_per_epoch == 0:
        sys.exit(f"ERROR: only {len(train_set)} training images: fewer than one batch of {cfg.batch_size}.")
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)

    model = build_model(cfg.model, cfg.num_classes, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(cfg.encoder_lr, cfg.head_lr, cfg.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = SegLoss(cfg.ce_weight, cfg.lovasz_weight)
    total_epochs, epoch_done, n_avg, avg = cfg.epochs, 0, 0, None

    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        diff = [k for k in settings if st["settings"].get(k) != settings[k]]
        if diff:
            sys.exit(f"ERROR: {run_dir} holds a fine-tuning run with different settings ({', '.join(diff)}). Use a new --run-name.")
        load_weights(model, st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        epoch_done, n_avg, avg = st["epoch_done"], st["n_avg"], st["avg"]
        print(f"[finetune] RESUMED after epoch {epoch_done} of {total_epochs}", flush=True)
    else:
        load_weights(model, init["model"])
        if average:  # the starting weights are the first member of the average
            avg, n_avg = {k: v.clone() for k, v in init["model"].items()}, 1
        for stale in (run_dir / "finetune_log.csv", run_dir / "cycles.csv"):
            stale.unlink(missing_ok=True)
        print(f"[finetune] FRESH START from {a.init_checkpoint}", flush=True)
    del init

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[finetune] {cfg.model} | {cfg.height}x{cfg.width} | batch {cfg.batch_size} | fp16 {amp} | {gpu} | {workers} workers | "
          f"{a.cycles} cycles x {a.epochs_per_cycle} epochs | lr x{a.lr_max_scale:g} -> x{a.lr_min_scale:g} per cycle | "
          + ("weight averaging ON" if average else "weight averaging OFF"), flush=True)
    print(f"[finetune] data: train {len(train_set)}, val {len(val_set)} images | {steps_per_epoch} steps/epoch", flush=True)

    eval_model = build_model(cfg.model, cfg.num_classes, pretrained=False).to(device) if average else None
    cycle_steps = a.epochs_per_cycle * steps_per_epoch
    last_metrics = None

    def append(path: Path, row: dict) -> None:
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def averaged_metrics() -> dict:
        eval_model.load_state_dict(avg)
        sampler.set_epoch(5 * EPOCH_OFFSET + epoch_done, 0)
        update_bn(eval_model, train_loader, device, amp, a.bn_batches)
        return validate(eval_model, val_loader, criterion, cfg.num_classes, device, amp, chunk=cfg.batch_size)

    for epoch in range(epoch_done, total_epochs):
        cycle, epoch_in_cycle = divmod(epoch, a.epochs_per_cycle)
        sampler.set_epoch(EPOCH_OFFSET + epoch, 0)
        model.train()
        seen, loss_sum, t0 = 0, 0.0, time.time()
        for b, batch in enumerate(train_loader):
            position = (epoch_in_cycle * steps_per_epoch + b) / cycle_steps  # 0 at the cycle start, 1 at its end
            factor = a.lr_min_scale + 0.5 * (a.lr_max_scale - a.lr_min_scale) * (1 + math.cos(math.pi * position))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
            loss, _ = criterion(logits, masks)
            if amp and not math.isfinite(float(loss.detach())):
                loss, _ = criterion(model(images), masks)  # rare fp16 overflow: redo in float32
            if not math.isfinite(float(loss.detach())):
                sys.exit(f"ERROR: loss became {float(loss)} in epoch {epoch + 1}. The saved state is intact; lower --lr-max-scale and use a new --run-name.")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            seen += 1
            loss_sum += float(loss.detach())
            if seen % a.log_every == 0 or b + 1 == steps_per_epoch:
                ips = seen * cfg.batch_size / max(1e-6, time.time() - t0)
                print(f"[finetune] cycle {cycle + 1}/{a.cycles} epoch {epoch_in_cycle + 1}/{a.epochs_per_cycle} step {b + 1}/{steps_per_epoch} | "
                      f"loss {loss_sum / seen:.4f} | lr x{factor:.3f} | {ips:.1f} img/s", flush=True)
        append(run_dir / "finetune_log.csv", {"epoch": epoch + 1, "cycle": cycle + 1, "train_loss": round(loss_sum / max(1, seen), 5)})
        epoch_done = epoch + 1

        if epoch_in_cycle == a.epochs_per_cycle - 1:  # end of a cycle
            current = validate(model, val_loader, criterion, cfg.num_classes, device, amp, chunk=cfg.batch_size)
            row = {"cycle": cycle + 1, "epoch": epoch_done, "cur_miou": round(current["miou"], 5), "cur_garment_miou": round(current["garment_miou"], 5)}
            text = f"[cycle {cycle + 1}/{a.cycles}] current weights: mIoU {current['miou']:.4f}, garment mIoU {current['garment_miou']:.4f}"
            last_metrics = current
            if average:
                n_avg += 1
                for k, v in model.state_dict().items():
                    v = v.detach().cpu()
                    avg[k] = avg[k] + (v - avg[k]) / n_avg if v.is_floating_point() else v.clone()
                averaged = averaged_metrics()
                row.update({"avg_miou": round(averaged["miou"], 5), "avg_garment_miou": round(averaged["garment_miou"], 5), "models_averaged": n_avg})
                text += f" | average of {n_avg} models: mIoU {averaged['miou']:.4f}, garment mIoU {averaged['garment_miou']:.4f}"
                last_metrics = averaged
            append(run_dir / "cycles.csv", row)
            print(text, flush=True)

        _save({"settings": settings, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
               "epoch_done": epoch_done, "n_avg": n_avg, "avg": avg}, state_path)
        if epoch_done < total_epochs and (time.time() - start_time) / 60 >= a.time_budget_min:
            print(f"[finetune] PAUSED after epoch {epoch_done} of {total_epochs} (time budget). Run the same command again to continue.", flush=True)
            return

    # Final model: the average over the cycles (with recomputed BatchNorm statistics), or the last weights.
    if average:
        if last_metrics is None:
            last_metrics = averaged_metrics()
        else:
            eval_model.load_state_dict(avg)
            sampler.set_epoch(5 * EPOCH_OFFSET + epoch_done, 0)
            update_bn(eval_model, train_loader, device, amp, a.bn_batches)
        final_state = eval_model.state_dict()
    else:
        if last_metrics is None:
            last_metrics = validate(model, val_loader, criterion, cfg.num_classes, device, amp, chunk=cfg.batch_size)
        final_state = model.state_dict()
    out = run_dir / "checkpoints" / "best.pt"
    _save({"model": final_state, "config": cfg.to_dict(), "epoch": total_epochs, "global_step": total_epochs * steps_per_epoch,
           "best_metric": last_metrics["miou"], "val_metrics": last_metrics}, out)
    print(f"[finetune] FINISHED. Final model ({'average of ' + str(n_avg) + ' models' if average else 'last weights'}): "
          f"val mIoU {last_metrics['miou']:.4f}, garment mIoU {last_metrics['garment_miou']:.4f} at {cfg.height}x{cfg.width}. Saved {out}", flush=True)


if __name__ == "__main__":
    main()
