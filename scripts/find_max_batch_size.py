"""Binary-search for the largest batch size that fits in VRAM budget (Step 2, Fix 4).

Runs 5 training steps at each candidate batch size, measures peak VRAM usage,
and returns the largest size that keeps VRAM utilisation below --vram-budget
(default 85%).

Usage:
    python scripts/find_max_batch_size.py --dataset fashionmnist
    python scripts/find_max_batch_size.py --dataset cifar10 --vram-budget 80
    python scripts/find_max_batch_size.py --start 32 --max 512 --steps 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _probe_batch_size(
    dataset_name: str,
    batch_size: int,
    steps: int,
    device,
) -> dict:
    """Run `steps` training steps at `batch_size` and return VRAM stats."""
    import torch
    from its.models import ScoreUNetConfig, build_score_model
    from its.training.controlled_score_training import (
        ControlledScoreTrainingConfig, _make_dataloader, _sample_sigma,
    )
    from its.training.score_training import _sample_sigma as _ss

    cfg = ControlledScoreTrainingConfig(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=0,
        dataset_subset_size=batch_size * (steps + 2),
        epochs=1,
        model=ScoreUNetConfig(in_channels=1 if dataset_name != "cifar10" else 3,
                               base_channels=64, channel_mults=(1, 2, 2)),
    )
    loader = _make_dataloader(cfg)
    model = build_score_model(cfg.model).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)

    torch.cuda.reset_peak_memory_stats(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory

    model.train()
    step = 0
    for batch in loader:
        if step >= steps:
            break
        x, _ = batch
        x = x.to(device)
        sigma = _ss(x, 0.01, 1.0)
        noise = torch.randn_like(x)
        noisy = x + sigma * noise
        target = -noise / sigma
        with torch.amp.autocast("cuda", enabled=True):
            score = model(noisy, sigma)
            loss = torch.mean((score - target) ** 2)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step += 1

    peak = torch.cuda.max_memory_allocated(device)
    util_pct = 100.0 * peak / total_vram
    del model, optim, loader
    torch.cuda.empty_cache()
    return {"batch_size": batch_size, "peak_bytes": peak, "total_bytes": total_vram,
            "util_pct": util_pct, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashionmnist",
                        choices=["fashionmnist", "cifar10", "mnist"])
    parser.add_argument("--start", type=int, default=32, help="Starting batch size")
    parser.add_argument("--max", type=int, default=1024, help="Max batch size to try")
    parser.add_argument("--steps", type=int, default=5, help="Steps per probe")
    parser.add_argument("--vram-budget", type=float, default=85.0,
                        help="Max VRAM utilisation %% (default: 85)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="Save result JSON to this path")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Batch size search requires a GPU.")
        sys.exit(1)

    device = torch.device(args.device or "cuda")
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(device)} ({total_gb:.1f} GB)")
    print(f"Searching batch sizes {args.start}..{args.max} with {args.vram_budget:.0f}% VRAM budget")

    results = []
    lo, hi = args.start, args.max
    best = args.start

    # First test start to ensure it fits
    try:
        r = _probe_batch_size(args.dataset, args.start, args.steps, device)
        results.append(r)
        print(f"  batch={args.start}: VRAM={r['util_pct']:.1f}%", end="")
        if r["util_pct"] <= args.vram_budget:
            print(" OK")
            best = args.start
        else:
            print(" TOO LARGE — start batch size already over budget!")
            sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print(f"  batch={args.start}: OOM")
        sys.exit(1)

    # Binary search
    lo = args.start * 2
    while lo <= hi:
        mid = (lo + hi) // 2
        mid = (mid // args.start) * args.start  # snap to multiple of start
        if mid == 0:
            break
        try:
            r = _probe_batch_size(args.dataset, mid, args.steps, device)
            results.append(r)
            print(f"  batch={mid}: VRAM={r['util_pct']:.1f}%", end="")
            if r["util_pct"] <= args.vram_budget:
                print(" OK — trying larger")
                best = mid
                lo = mid + args.start
            else:
                print(" over budget — trying smaller")
                hi = mid - args.start
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={mid}: OOM — trying smaller")
            hi = mid - args.start

    # Linear scaling rule for LR
    old_lr = 2e-4
    new_lr = old_lr * (best / args.start)
    print(f"\nResult: max batch size = {best}")
    print(f"  Linear scaling rule: LR {old_lr:.2e} -> {new_lr:.2e}")

    summary = {
        "dataset": args.dataset,
        "recommended_batch_size": best,
        "vram_budget_pct": args.vram_budget,
        "recommended_lr": new_lr,
        "old_lr": old_lr,
        "probes": results,
    }
    out_path = args.out or str(ROOT / "data" / "results" / "max_batch_size.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
