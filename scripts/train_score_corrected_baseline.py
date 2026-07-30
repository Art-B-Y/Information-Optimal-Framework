"""Phase 2 Step 1 — retrain the canonical baseline score model with the CORRECTED pipeline.

Every pre-Phase-2 checkpoint is void: they were trained under a pipeline with four
fatal defects (see docs/current_state_diagnosis.md) and three further training defects
found in Phase 2 pre-flight (data/results/phase2_preflight.txt). This run produces the
new canonical baseline.

What is corrected here relative to every prior run:
  1. VE training kernel  <->  VE sampler          (was: VE-trained, VP-sampled -- defect A2)
  2. sigma_max = 42.0                             (was: 1.0, ~42x too small for a VE prior)
  3. dsm_weighting = "sigma_sq"                   (was: unweighted; gradient collapsed onto small sigma)
  4. preconditioning = "edm"                      (was: h = x/sigma, 70x input-scale range)
  5. AMP DISABLED                                 (GTX 1650 Ti / TU117 has no tensor cores; AMP is 3.7x SLOWER)

Usage:
    python scripts/train_score_corrected_baseline.py
    python scripts/train_score_corrected_baseline.py --epochs 100 --batch-size 256
    # via the fault-tolerant segmented launcher:
    python scripts/launch_segmented_training.py --script scripts/train_score_corrected_baseline.py \
        --checkpoint-dir checkpoints/fmnist_score_corrected_baseline --segment-hours 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_NAME = "fmnist_score_corrected_baseline"

# Song's Technique 1: sigma_max = max pairwise distance between training points.
# Measured on FashionMNIST normalised to [-1,1]: 41.88 (see phase2_preflight.txt).
SIGMA_MIN = 0.01
SIGMA_MAX = 42.0
SIGMA_DATA = 0.70  # measured FashionMNIST std in [-1,1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run", default=None, help="Ignored; accepted for launcher compatibility.")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from the newest checkpoint in the run dir.")
    parser.add_argument("--max-wall-hours", type=float, default=0.0,
                        help="Exit with code 42 after this many hours (segmented launcher).")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This run must NOT fall back to CPU -- a CPU-only "
            "torch build is what turned run 5A into a 176-hour job. Fix the environment first."
        )

    ckpt_dir = ROOT / "checkpoints" / RUN_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_latest and args.resume_from is None:
        last = ckpt_dir / "score_last.pt"
        auto = ckpt_dir / "autosave.pt"
        candidates = sorted(ckpt_dir.glob("score_epoch_*.pt"))
        picks = [p for p in (auto, last) if p.exists()]
        if candidates:
            picks.append(candidates[-1])
        if picks:
            args.resume_from = str(max(picks, key=lambda p: p.stat().st_mtime))
            print(f"--resume-latest: resuming from {args.resume_from}")

    from its.training.score_training import ScoreTrainingConfig, train_score_model
    from its.models import ScoreUNetConfig
    from its.data import DatasetConfig

    jsonl_log = ROOT / "logs" / f"{RUN_NAME}.jsonl"
    jsonl_log.parent.mkdir(parents=True, exist_ok=True)

    cfg = ScoreTrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        dsm_weighting="sigma_sq",
        grad_clip=1.0,
        device=args.device or "cuda",
        log_interval=25,
        dataset=DatasetConfig(
            name="fashionmnist",
            batch_size=args.batch_size,
            num_workers=4,
        ),
        model=ScoreUNetConfig(
            in_channels=1,
            base_channels=args.base_channels,
            channel_mults=(1, 2, 2),
            use_time_embedding=True,
            time_embed_dim=128,
            use_attention=False,
            preconditioning="edm",
            sigma_data=SIGMA_DATA,
        ),
        ema_decay=0.9999,
        checkpoint_dir=str(ckpt_dir),
        save_interval=5,
        resume_from=args.resume_from,
        log_dir=str(ROOT / "logs"),
        run_name=RUN_NAME,
        nan_tolerance=5,
        jsonl_log=str(jsonl_log),
        use_lr_schedule=True,
        lr_min=1e-6,
        val_every_n_epochs=1,
        log_sample_diversity=False,
        # AMP is 3.7x SLOWER on this GPU (TU117, no tensor cores). Measured, not assumed.
        mixed_precision=False,
        max_wall_hours=args.max_wall_hours,
    )

    print(f"=== {RUN_NAME} ===")
    print(f"  convention   : VE kernel (x + sigma*eps) <-> VE sampler   [MATCHED]")
    print(f"  sigma range  : [{SIGMA_MIN}, {SIGMA_MAX}]  (Song technique-1)")
    print(f"  weighting    : sigma_sq   preconditioning: edm   AMP: off")
    print(f"  epochs={args.epochs} batch={args.batch_size} lr={args.lr} bc={args.base_channels}")
    print(f"  checkpoints -> {ckpt_dir}")
    print(f"  jsonl       -> {jsonl_log}")

    metrics = train_score_model(cfg)

    out = ROOT / "data" / "results" / f"{RUN_NAME}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nTraining complete. Metrics: {metrics}")
    print(f"Metrics saved -> {out}")


if __name__ == "__main__":
    main()
