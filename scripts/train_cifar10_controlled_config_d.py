"""Controlled Config D training on CIFAR-10.

Step 2D (Session 7). Uses the best cifar10_score_v2 checkpoint as frozen backbone.
Trains for 50 epochs with path_kl + quality terms.

Usage:
    python scripts/train_cifar10_controlled_config_d.py --score-ckpt checkpoints/score_cifar10_v2/score_best.pt
    python scripts/train_cifar10_controlled_config_d.py --score-ckpt <ckpt> --epochs 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_one_seed(score_ckpt: str, epochs: int, seed: int, device: str | None,
                 num_samples_eval: int = 2048,
                 resume_latest: bool = False,
                 max_wall_hours: float = 0.0) -> dict:
    import torch
    from its.training.controlled_score_training import ControlledScoreTrainingConfig, train_controlled_score
    from its.models import ScoreUNetConfig
    from its.controllers import ControlConfig
    from its.sde import ScoreSDEConfig

    torch.manual_seed(seed)

    ckpt_dir = ROOT / "checkpoints" / f"cifar10_controlled_d_seed{seed}"
    jsonl_log = ROOT / "logs" / f"cifar10_controlled_d_seed{seed}.jsonl"
    jsonl_log.parent.mkdir(parents=True, exist_ok=True)

    cfg = ControlledScoreTrainingConfig(
        epochs=epochs,
        batch_size=32,
        dataset_name="cifar10",
        device=device,
        lr=2e-4,
        control_weight=0.01,
        path_kl_weight=0.1,
        quality_weight=0.01,
        grad_clip=1.0,
        log_interval=100,
        nan_tolerance=10,
        jsonl_log=str(jsonl_log),
        checkpoint_dir=str(ckpt_dir),
        save_interval=5,
        log_dir=str(ROOT / "logs"),
        run_name=f"cifar10_controlled_d_seed{seed}",
        use_lr_schedule=True,
        lr_min=1e-6,
        ema_decay=0.999,
        score_backbone_ckpt=score_ckpt,
        freeze_score_model=True,
        mixed_precision=True,
        dataset_subset_size=5000,
        resume_latest=resume_latest,
        checkpoint_every_n_minutes=30,
        max_wall_hours=max_wall_hours,
        sde=ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=10, clamp=5.0, control_weight=1.0),
        model=ScoreUNetConfig(
            in_channels=3,
            base_channels=64,
            channel_mults=(1, 2, 2, 4),
            use_time_embedding=True,
            time_embed_dim=128,
            use_attention=True,
        ),
        control=ControlConfig(
            state_dim=3072,  # 3 x 32 x 32
            hidden_dim=512,
            depth=4,
            time_embedding_dim=16,
        ),
    )

    print(f"  CIFAR-10 Config D seed={seed}: {epochs} epochs, backbone={score_ckpt}")
    metrics = train_controlled_score(cfg)
    return {"seed": seed, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run", default=None,
                        help="Run ID passed by the segmented launcher (ignored).")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from autosave.pt or controlled_last.pt (used by segmented launcher).")
    parser.add_argument("--max-wall-hours", type=float, default=0.0,
                        help="Exit with code 42 after this many wall-clock hours (used by segmented launcher).")
    args = parser.parse_args()

    result = run_one_seed(args.score_ckpt, args.epochs, args.seed, args.device,
                          resume_latest=args.resume_latest,
                          max_wall_hours=args.max_wall_hours)

    out_path = ROOT / "data" / "results" / "cifar10_controlled_d_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": "D", "dataset": "cifar10",
                                    "epochs": args.epochs, "runs": [result]}, indent=2))
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
