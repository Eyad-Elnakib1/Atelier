"""Measure training speed and GPU memory of each candidate model on this GPU.

Uses random tensors at the training resolution, so it only measures the GPU side
(no data loading). The numbers replace the estimates in PLAN.md and decide how
many epochs fit in the schedule.

Example (Colab):
    python scripts/benchmark.py --out /content/drive/MyDrive/clothseg/reports/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

CANDIDATES = ("segformer_b2", "segformer_b3", "segformer_b4", "upernet_convnext_tiny")


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    """Build a candidate with ImageNet-pretrained encoder weights."""
    if name.startswith("segformer_"):
        from transformers import SegformerForSemanticSegmentation
        return SegformerForSemanticSegmentation.from_pretrained(f"nvidia/mit-{name.split('_')[1]}", num_labels=num_classes)
    if name == "upernet_convnext_tiny":
        import segmentation_models_pytorch as smp
        return smp.UPerNet(encoder_name="tu-convnext_tiny", encoder_weights="imagenet", classes=num_classes)
    raise ValueError(f"unknown model '{name}'")


def forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the model and return logits at the input resolution."""
    out = model(x)
    logits = out.logits if hasattr(out, "logits") else out
    if logits.shape[-2:] != x.shape[-2:]:
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def bench(name: str, args: argparse.Namespace, device: torch.device) -> dict:
    model = build_model(name, args.num_classes).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=6e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    x = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    y = torch.randint(0, args.num_classes, (args.batch_size, args.height, args.width), device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            sync()
            start = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = F.cross_entropy(forward_logits(model, x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    sync()
    train_ips = args.steps * args.batch_size / (time.time() - start)
    train_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0

    model.eval()
    x1 = x[:1]
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for _ in range(5):
            forward_logits(model, x1)
        sync()
        start = time.time()
        for _ in range(30):
            forward_logits(model, x1)
        sync()
    infer_ms = 1000 * (time.time() - start) / 30

    del model, opt, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": name, "params_M": round(params, 1), "train_img_per_s": round(train_ips, 1),
        "min_per_epoch": round(args.train_images / train_ips / 60, 1), "train_vram_GB": round(train_mem, 2),
        "infer_ms_per_img": round(infer_ms, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=CANDIDATES)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=15)
    p.add_argument("--steps", type=int, default=30, help="Timed training steps per model.")
    p.add_argument("--warmup", type=int, default=5, help="Untimed warm-up steps per model.")
    p.add_argument("--train-images", type=int, default=16000, help="Used to convert speed into minutes per epoch.")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON file for the results.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (no GPU found: numbers are not meaningful)"
    print(f"[benchmark] device: {gpu} | torch {torch.__version__} | batch {args.batch_size} at {args.height}x{args.width} | fp16: {device.type == 'cuda'}", flush=True)

    results = []
    for name in args.models:
        print(f"[benchmark] {name} ...", flush=True)
        try:
            results.append(bench(name, args, device))
            print(f"[benchmark]   {results[-1]}", flush=True)
        except torch.cuda.OutOfMemoryError:
            results.append({"model": name, "error": f"out of GPU memory at batch {args.batch_size}"})
            print(f"[benchmark]   OUT OF MEMORY at batch {args.batch_size}. Rerun with --batch-size 8 for this model.", flush=True)
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - keep benchmarking the other models
            results.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[benchmark]   FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    print(f"\n{'model':<24}{'params M':>9}{'img/s':>8}{'min/epoch':>11}{'VRAM GB':>9}{'infer ms':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['model']:<24}  {r['error']}")
        else:
            print(f"{r['model']:<24}{r['params_M']:>9}{r['train_img_per_s']:>8}{r['min_per_epoch']:>11}{r['train_vram_GB']:>9}{r['infer_ms_per_img']:>10}")
    print("(min/epoch counts GPU work only; real epochs are slower if the 2 CPU cores cannot feed images fast enough)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"gpu": gpu, "torch": torch.__version__, "args": {k: str(v) for k, v in vars(args).items()}, "results": results}, indent=2))
        print(f"[benchmark] saved {args.out}")


if __name__ == "__main__":
    main()
