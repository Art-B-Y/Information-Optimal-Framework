"""Train CIFAR-10 score model v2 — 50 epochs, GPU, time embedding + attention.

Session 6, Step 2A. Uses cosine annealing LR, validation loss every 5 epochs.
Saves checkpoints every 10 epochs to checkpoints/score_cifar10_v2/.

Note: GTX 1650 Ti (4GB) can handle base_channels=64 with batch=128.
For base_channels=128 reduce batch to 64.

Usage:
    python scripts/train_cifar10_score_v2.py
    python scripts/train_cifar10_score_v2.py --epochs 50 --batch-size 64 --base-channels 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None,
                        help="Path to a specific checkpoint to resume from.")
    parser.add_argument("--use-attention", action="store_true", default=True)
    parser.add_argument("--run", default=None,
                        help="Run ID passed by the segmented launcher (ignored by this script).")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from the most recent score_epoch_*.pt in the checkpoint dir.")
    parser.add_argument("--max-wall-hours", type=float, default=0.0,
                        help="Exit with code 42 after this many wall-clock hours (used by segmented launcher).")
    args = parser.parse_args()

    from its.training.score_training import ScoreTrainingConfig, train_score_model
    from its.models import ScoreUNetConfig
    from its.data import DatasetConfig

    ckpt_dir = ROOT / "checkpoints" / "score_cifar10_v2"

    # --resume-latest: prefer score_last.pt (wall-clock exit point) when it is
    # newer than the latest score_epoch_*.pt; otherwise use the latest epoch ckpt.
    if args.resume_latest and args.resume_from is None:
        last = ckpt_dir / "score_last.pt"
        candidates = sorted(ckpt_dir.glob("score_epoch_*.pt"))
        if last.exists() and (not candidates or last.stat().st_mtime > candidates[-1].stat().st_mtime):
            args.resume_from = str(last)
        elif candidates:
            args.resume_from = str(candidates[-1])
        elif last.exists():
            args.resume_from = str(last)
        if args.resume_from:
            print(f"--resume-latest: resuming from {args.resume_from}")

    jsonl_log = ROOT / "logs" / "cifar10_score_v2.jsonl"
    jsonl_log.parent.mkdir(parents=True, exist_ok=True)

    cfg = ScoreTrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        sigma_min=0.01,
        sigma_max=1.0,
        grad_clip=1.0,
        device=args.device,
        log_interval=100,
        dataset=DatasetConfig(
            name="cifar10",
            batch_size=args.batch_size,
            num_workers=4,
        ),
        model=ScoreUNetConfig(
            in_channels=3,
            base_channels=args.base_channels,
            channel_mults=(1, 2, 2, 4),
            use_time_embedding=True,
            time_embed_dim=128,
            use_attention=True,
            attention_heads=4,
        ),
        ema_decay=0.999,
        checkpoint_dir=str(ckpt_dir),
        save_interval=10,
        resume_from=args.resume_from,
        log_dir=str(ROOT / "logs"),
        run_name="cifar10_score_v2",
        nan_tolerance=5,
        jsonl_log=str(jsonl_log),
        use_lr_schedule=True,
        lr_min=1e-6,
        val_every_n_epochs=5,
        log_sample_diversity=False,
        max_wall_hours=args.max_wall_hours,
    )

    print(f"Training CIFAR-10 score v2: {args.epochs} epochs, "
          f"base_channels={args.base_channels}, batch={args.batch_size}, "
          f"device={args.device or 'auto'}")
    print(f"Checkpoints -> {ckpt_dir}")
    print(f"JSONL log    -> {jsonl_log}")

    metrics = train_score_model(cfg)

    result_path = ROOT / "data" / "results" / "cifar10_score_v2_metrics.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nTraining complete. Metrics: {metrics}")
    print(f"Metrics saved -> {result_path}")


if __name__ == "__main__":
    main()
