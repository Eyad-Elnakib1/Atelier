"""Training and validation loops with crash-proof resume.

How a run survives a dead session
---------------------------------
* Everything persistent lives in one run directory (normally on Google Drive):
  ``config.json``, ``train_log.csv``, ``val_log.csv`` and ``checkpoints/``.
* A checkpoint is written every ``save_every_min`` minutes, at the end of every
  epoch, and when the time budget is reached. It holds the model, optimizer and
  GradScaler states, the position (epoch, batch inside the epoch, global step), the
  best metric, the random-generator states and the config.
* The learning rate is a pure function of the global step, so there is no scheduler
  object whose state could be lost.
* The data order and every augmentation are pure functions of (seed, epoch,
  position), see ``data/dataset.py``. A resumed run therefore continues with exactly
  the batches it would have seen. (GPU arithmetic itself is not bit-reproducible, so
  losses after a resume match an uninterrupted run closely, not bit for bit.)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_latest, rng_state, save_best, save_checkpoint, set_rng_state
from .config import TrainConfig
from .data.dataset import AtrDataset, ResumableBatchSampler
from .data.splits import load_splits
from .data.storage import atomic_copy
from .labels import class_names
from .losses import SegLoss
from .metrics import ConfusionMatrix
from .models import SegModel, build_model, load_weights

TRAIN_FIELDS = ["step", "epoch", "batch", "loss", "ce", "lovasz", "lr_encoder", "lr_head", "img_per_s", "train_min", "batch_hash"]


@dataclass
class Runtime:
    """Session settings that do not define the experiment."""
    run_dir: Path
    cache_dir: Path
    workers: int = 2
    time_budget_min: float = 280.0   # stop cleanly before Colab ends the session
    save_every_min: float = 10.0
    log_every: int = 50
    stop_after_epoch: int = 0        # >0: pause after this many epochs (bake-off rounds)
    stop_after_steps: int = 0        # >0: pause after this global step (resume test)
    extend_from: Path | None = None  # fresh start only: continue this finished run with a longer schedule


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warm-up multiplied by a linear ("poly", power 1) decay to zero."""
    warm = min(1.0, (step + 1) / max(1, warmup_steps))
    return warm * max(0.0, 1.0 - step / max(1, total_steps))


def data_fingerprint(cache_dir: Path) -> str:
    """Short hash of the split membership; changes if the dataset or the splits change."""
    splits = load_splits(cache_dir)
    text = json.dumps({k: splits[k] for k in ("train", "val", "test")}, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _truncate_csv(path: Path, max_step: int) -> None:
    """Drop log rows written after the checkpoint we resume from, so curves have no duplicates."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, [r for r in reader if int(r["step"]) <= max_step]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model: SegModel, loader: DataLoader, criterion: SegLoss, num_classes: int,
             device: torch.device, amp: bool, chunk: int = 0) -> dict:
    """Run the model over a loader and return the metrics of ``ConfusionMatrix.compute`` plus the loss.

    ``chunk`` > 0 passes each validation batch through the model in pieces of at most
    that many images. The loss and the metrics are still computed on the whole batch,
    so the numbers do not change; only the peak GPU memory is lower.
    """
    model.eval()
    if device.type == "cuda":
        # Training leaves its memory reserved in blocks sized for training batches. Release
        # them so validation, which uses larger batches, does not run out on a 16 GB GPU.
        torch.cuda.empty_cache()

    def forward(images: torch.Tensor, use_amp: bool) -> torch.Tensor:
        parts = images.split(chunk) if chunk > 0 else (images,)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            return torch.cat([model(part) for part in parts])

    cm = ConfusionMatrix(num_classes, device)
    loss_sum, n, fp32_batches = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = forward(images, amp)
        if amp and not torch.isfinite(logits).all():
            # Some GPUs occasionally overflow in half precision; redo this batch in float32.
            logits = forward(images, False)
            fp32_batches += 1
        loss, _ = criterion(logits, masks)
        loss_sum += float(loss) * images.shape[0]
        n += images.shape[0]
        cm.update(logits.argmax(1), masks)
    if fp32_batches:
        print(f"[val] note: {fp32_batches} batch(es) gave non-finite fp16 outputs and were recomputed in float32", flush=True)
    metrics = cm.compute()
    metrics["val_loss"] = loss_sum / max(1, n)
    return metrics


def train(cfg: TrainConfig, rt: Runtime) -> str:
    """Train or resume a run. Returns ``"finished"``, ``"paused"`` or ``"nothing_to_do"``."""
    session_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"
    run_dir = Path(rt.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_csv, val_csv = run_dir / "train_log.csv", run_dir / "val_log.csv"
    names = class_names(cfg.num_classes)
    val_fields = ["epoch", "step", "miou", "garment_miou", "clothing_binary_iou", "pixel_acc", "mean_class_acc",
                  "mean_dice", "val_loss", "train_min"] + [f"iou_{n}" for n in names]

    cfg.data_fingerprint = data_fingerprint(rt.cache_dir)
    state = load_latest(run_dir)
    extended_from = None
    if state is None and rt.extend_from is not None:
        # Continue a finished run with a longer schedule: weights, optimizer state, step counter
        # and random state are taken from its last checkpoint, so the new run behaves as if that
        # run had been started with the longer schedule, except that its learning rate had
        # already fallen to zero and now steps back up to where the longer schedule would be.
        source_dir = Path(rt.extend_from)
        state = load_latest(source_dir)
        if state is None:
            raise RuntimeError(f"--extend-from: no readable checkpoint in {source_dir}.")
        source = TrainConfig.from_dict(state["config"])
        diff = [d for d in source.differences(cfg) if not d.startswith("epochs:")]
        if diff:
            raise RuntimeError(f"--extend-from: {source_dir} was trained with other settings:\n  " + "\n  ".join(diff))
        if state["epoch"] != source.epochs or state["batch_in_epoch"] != 0:
            raise RuntimeError(f"--extend-from: {source_dir} has not finished its {source.epochs}-epoch schedule.")
        if cfg.epochs <= source.epochs:
            raise RuntimeError(f"--extend-from: --epochs must be larger than the {source.epochs} epochs of {source_dir}.")
        state["config"] = cfg.to_dict()
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        if (source_dir / "checkpoints" / "best.pt").exists():  # the best model so far stays the best until beaten
            atomic_copy(source_dir / "checkpoints" / "best.pt", run_dir / "checkpoints" / "best.pt", quiet=True)
        extended_from = (source_dir.name, source.epochs)
    config_path = run_dir / "config.json"
    if state is not None:
        saved = TrainConfig.from_dict(state["config"])
        diff = saved.differences(cfg)
        if diff:
            raise RuntimeError(
                f"The run directory {run_dir} already holds a run with different settings:\n  "
                + "\n  ".join(diff)
                + "\nUse a new --run-name for a new experiment, or repeat the original settings to resume this one."
            )
    elif config_path.exists() and TrainConfig.from_dict(json.loads(config_path.read_text())).differences(cfg):
        print(f"[train] WARNING: {config_path} is from an earlier attempt with other settings and no checkpoint. Overwriting it.", flush=True)
    cfg.save(config_path)

    # Seeds only matter for a fresh start; a resumed run restores the saved generator states below.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    train_set = AtrDataset(rt.cache_dir, "train", cfg.num_classes, cfg.height, cfg.width, train=True, seed=cfg.seed, limit=cfg.limit_train or None, augment=cfg.augment)
    val_set = AtrDataset(rt.cache_dir, "val", cfg.num_classes, cfg.height, cfg.width, train=False, limit=cfg.limit_val or None)
    sampler = ResumableBatchSampler(len(train_set), cfg.batch_size, cfg.seed)
    steps_per_epoch = sampler.batches_per_epoch
    if steps_per_epoch == 0:
        raise RuntimeError(f"Only {len(train_set)} training images: fewer than one batch of {cfg.batch_size}.")
    total_steps = cfg.epochs * steps_per_epoch
    loader_gen = torch.Generator()  # keeps DataLoader worker seeding away from the global generator
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=rt.workers, pin_memory=device.type == "cuda",
                              persistent_workers=rt.workers > 0, generator=loader_gen)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=rt.workers,
                            pin_memory=device.type == "cuda", persistent_workers=rt.workers > 0)

    model = build_model(cfg.model, cfg.num_classes, pretrained=cfg.pretrained and state is None).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(cfg.encoder_lr, cfg.head_lr, cfg.weight_decay), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = SegLoss(cfg.ce_weight, cfg.lovasz_weight)

    epoch = batch_in_epoch = global_step = 0
    best_metric, best_epoch, train_seconds = -1.0, 0, 0.0
    if state is not None:
        load_weights(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch, batch_in_epoch, global_step = state["epoch"], state["batch_in_epoch"], state["global_step"]
        best_metric, best_epoch = state["best_metric"], state.get("best_epoch", 0)
        train_seconds = state.get("train_seconds", 0.0)
        set_rng_state(state["rng"])
        _truncate_csv(train_csv, global_step)
        _truncate_csv(val_csv, global_step)
        if extended_from:
            print(f"[train] EXTENDED: continues run {extended_from[0]} after its epoch {extended_from[1]}, now with a {cfg.epochs}-epoch schedule. "
                  f"Learning rate restarts at x{lr_factor(global_step, total_steps, cfg.warmup_steps):.3f} of the base rate and falls to zero at epoch {cfg.epochs}. "
                  f"best.pt starts as the best model of {extended_from[0]}.", flush=True)
        print(f"[train] RESUMED from epoch {epoch + 1}, batch {batch_in_epoch}/{steps_per_epoch} (global step {global_step}/{total_steps}); "
              + (f"best mIoU so far {best_metric:.4f}" if best_metric >= 0 else "no validation yet")
              + f"; {train_seconds / 60:.0f} min trained in earlier sessions; model weights restored from the checkpoint", flush=True)
        del state
    else:
        for stale in (train_csv, val_csv):  # logs of an attempt that never reached a checkpoint
            stale.unlink(missing_ok=True)
        print(f"[train] FRESH START in {run_dir}", flush=True)

    last_epoch = min(cfg.epochs, rt.stop_after_epoch) if rt.stop_after_epoch > 0 else cfg.epochs
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[train] {cfg.model} | {cfg.num_classes} classes | {cfg.height}x{cfg.width} | batch {cfg.batch_size} | fp16 {amp} | {gpu} | {rt.workers} workers"
          + ("" if cfg.augment else " | NO AUGMENTATION"), flush=True)
    print(f"[train] data: train {len(train_set)}, val {len(val_set)} images (fingerprint {cfg.data_fingerprint}) | {steps_per_epoch} steps/epoch", flush=True)
    print(f"[train] schedule: {cfg.epochs} epochs = {total_steps} steps; this command runs until the end of epoch {last_epoch}; "
          f"time budget {rt.time_budget_min:g} min; checkpoint every {rt.save_every_min:g} min", flush=True)
    if epoch >= last_epoch:
        print(f"[train] Nothing to do: epoch {last_epoch} is already complete for this run.", flush=True)
        return "nothing_to_do"

    def snapshot(val_metrics: dict | None = None) -> dict:
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "batch_in_epoch": batch_in_epoch, "global_step": global_step,
            "best_metric": best_metric, "best_epoch": best_epoch, "train_seconds": train_seconds,
            "rng": rng_state(), "config": cfg.to_dict(), "val_metrics": val_metrics,
        }

    def out_of_time() -> bool:
        return (time.time() - session_start) / 60 >= rt.time_budget_min

    def pause(reason: str) -> str:
        path = save_checkpoint(run_dir, snapshot())
        print(f"[train] PAUSED ({reason}) at epoch {epoch + 1}, batch {batch_in_epoch}/{steps_per_epoch}, step {global_step}. "
              f"Saved {path.name}. Run the same command again to continue.", flush=True)
        return "paused"

    last_save = time.time()
    fp32_retries = 0
    while epoch < last_epoch:
        sampler.set_epoch(epoch, batch_in_epoch)
        loader_gen.manual_seed(cfg.seed * 100003 + epoch)
        model.train()
        window = {"loss": 0.0, "ce": 0.0, "lovasz": 0.0, "n": 0, "t": time.time()}
        tick = time.time()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            factor = lr_factor(global_step, total_steps, cfg.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor

            def step_loss(use_amp: bool) -> tuple[torch.Tensor, dict]:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                loss, parts = criterion(logits, masks)
                extra = model.aux_loss(masks)  # e.g. the edge branch; None for plain models
                if extra is not None:
                    loss = loss + extra[0]
                    parts.update(extra[1])
                return loss, parts

            loss, parts = step_loss(amp)
            if amp and not math.isfinite(float(loss.detach())):
                # Half precision can overflow on a rare batch (and on some GPUs, e.g. GTX 16xx,
                # it is unreliable). Redo this one step in float32 before giving up.
                loss, parts = step_loss(False)
                fp32_retries += 1
                print(f"[train] note: step {global_step} gave a non-finite fp16 loss and was recomputed in float32 "
                      f"({fp32_retries} such step(s) this session)", flush=True)
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError(
                    f"Loss became {float(loss)} at step {global_step} (epoch {epoch + 1}). Nothing was saved from this step. "
                    "The last checkpoint is intact; rerun to resume from it. If it repeats, lower --encoder-lr/--head-lr."
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            batch_in_epoch += 1
            now = time.time()
            train_seconds += now - tick
            tick = now
            window["loss"] += float(loss.detach())
            window["ce"] += parts["ce"]
            window["lovasz"] += parts["lovasz"]
            window["n"] += 1

            if global_step % rt.log_every == 0 or batch_in_epoch == steps_per_epoch:
                n = window["n"]
                ips = n * cfg.batch_size / max(1e-6, now - window["t"])
                lrs = {g["name"].split("_")[0]: g["lr"] for g in optimizer.param_groups}
                row = {
                    "step": global_step, "epoch": epoch + 1, "batch": batch_in_epoch,
                    "loss": round(window["loss"] / n, 5), "ce": round(window["ce"] / n, 5), "lovasz": round(window["lovasz"] / n, 5),
                    "lr_encoder": f"{lrs.get('encoder', 0):.3e}", "lr_head": f"{lrs.get('head', 0):.3e}",
                    "img_per_s": round(ips, 1), "train_min": round(train_seconds / 60, 2),
                    "batch_hash": int(batch["idx"].sum()),  # identifies the batch: proves a resume replays the same data
                }
                _append_csv(train_csv, TRAIN_FIELDS, row)
                eta = (last_epoch * steps_per_epoch - global_step) * cfg.batch_size / max(1e-6, ips) / 60
                print(f"[train] epoch {epoch + 1}/{cfg.epochs} step {batch_in_epoch}/{steps_per_epoch} | loss {row['loss']:.4f} "
                      f"(ce {row['ce']:.4f}, lovasz {row['lovasz']:.4f}) | lr {lrs.get('encoder', 0):.2e} | {ips:.1f} img/s | "
                      f"about {eta:.0f} min left for this command (plus validation)", flush=True)
                window = {"loss": 0.0, "ce": 0.0, "lovasz": 0.0, "n": 0, "t": time.time()}

            if rt.stop_after_steps > 0 and global_step >= rt.stop_after_steps:
                return pause(f"--stop-after-steps {rt.stop_after_steps}")
            if out_of_time():
                return pause(f"time budget of {rt.time_budget_min:g} min reached")
            if (time.time() - last_save) / 60 >= rt.save_every_min:
                path = save_checkpoint(run_dir, snapshot())
                last_save = time.time()
                tick = time.time()  # do not count the save as training time
                print(f"[train] checkpoint {path.name} saved (step {global_step})", flush=True)

        # ---- end of epoch: validate, log, checkpoint ----
        t0 = time.time()
        metrics = validate(model, val_loader, criterion, cfg.num_classes, device, amp, chunk=cfg.batch_size)
        if device.type == "cuda":
            torch.cuda.empty_cache()  # hand the validation blocks back before training continues
        epoch += 1
        batch_in_epoch = 0
        is_best = metrics["miou"] > best_metric
        if is_best:
            best_metric, best_epoch = metrics["miou"], epoch
        row = {k: round(metrics[k], 5) for k in ("miou", "garment_miou", "clothing_binary_iou", "pixel_acc", "mean_class_acc", "mean_dice", "val_loss")}
        row.update({"epoch": epoch, "step": global_step, "train_min": round(train_seconds / 60, 2)})
        row.update({f"iou_{n}": round(v, 5) for n, v in metrics["iou"].items()})
        _append_csv(val_csv, val_fields, row)
        worst = sorted(((v, n) for n, v in metrics["iou"].items() if not math.isnan(v)))[:3]
        print(f"[val] epoch {epoch}/{cfg.epochs} | mIoU {metrics['miou']:.4f} | garment mIoU {metrics['garment_miou']:.4f} | "
              f"pixel acc {metrics['pixel_acc']:.4f} | val loss {metrics['val_loss']:.4f} | weakest: "
              + ", ".join(f"{n} {v:.2f}" for v, n in worst)
              + f" | {'NEW BEST' if is_best else f'best is {best_metric:.4f} (epoch {best_epoch})'} | validation took {time.time() - t0:.0f}s", flush=True)

        state_now = snapshot(metrics)
        path = save_checkpoint(run_dir, state_now)
        if is_best:
            save_best(run_dir, state_now)
        del state_now
        last_save = time.time()
        print(f"[train] checkpoint {path.name} saved" + (" and best.pt updated" if is_best else ""), flush=True)
        (run_dir / "status.json").write_text(json.dumps({
            "model": cfg.model, "epochs_done": epoch, "epochs_total": cfg.epochs, "global_step": global_step,
            "best_miou": best_metric, "best_epoch": best_epoch, "train_min": round(train_seconds / 60, 1),
        }, indent=2))

        if epoch < last_epoch and out_of_time():
            print(f"[train] PAUSED (time budget of {rt.time_budget_min:g} min reached) after epoch {epoch}. Run the same command again to continue.", flush=True)
            return "paused"

    done = "FINISHED" if epoch >= cfg.epochs else f"STOPPED AS REQUESTED after epoch {epoch} of {cfg.epochs}"
    print(f"[train] {done}. Best val mIoU {best_metric:.4f} at epoch {best_epoch}. Total training time {train_seconds / 60:.0f} min. Run directory: {run_dir}", flush=True)
    return "finished"
