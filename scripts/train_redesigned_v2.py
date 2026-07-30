"""Session 10 Step 5: Launch redesigned v2 controlled training runs.

Three runs:
  5A: Frozen backbone, v2 objective (detach CE, trajectory quality, REINFORCE warmup)
  5B: Joint fine-tune (score model unfrozen, distillation regularizer)
  5C: Path-KL ablation (path_kl_weight=0, v2 objective)

Usage:
    python scripts/train_redesigned_v2.py --run 5a
    python scripts/train_redesigned_v2.py --run 5b
    python scripts/train_redesigned_v2.py --run 5c
    python scripts/train_redesigned_v2.py --run all
    python scripts/train_redesigned_v2.py --run 5a --pilot   # fast pilot (5000 samples, 20 epochs)

NOTE ON RUNTIME: Full training (~60k samples, 100 epochs, SDE 20 steps) takes ~10-15h
on GTX 1650 Ti. Run with --pilot for a 30-min test. Reduce --sde-steps for speedup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from its.training.controlled_score_training import ControlledScoreTrainingConfig, train_controlled_score
from its.models import ScoreUNetConfig
from its.sde import ScoreSDEConfig
from its.controllers.neural_control import ConvControlConfig


# Shared architecture: matches Session 9 ConvControl
_SCORE_CFG = ScoreUNetConfig(
    in_channels=1,
    base_channels=64,
    channel_mults=(1, 2, 2),
    use_time_embedding=True,
    time_embed_dim=128,
)

_SDE_CFG = ScoreSDEConfig(
    beta_min=0.1,
    beta_max=10.0,
    num_steps=20,       # 20 steps (was 50 in Session 9); much faster, acceptable quality
    sigma_min=0.01,
    sigma_max=1.0,
    control_weight=1.0,
)

_CONV_CFG = ConvControlConfig(
    in_channels=1,
    base_channels=64,
    num_blocks=3,
    time_embed_dim=128,
    dropout=0.1,
)

_SCORE_CKPT = str(ROOT / "checkpoints" / "score_fmnist_v2" / "score_best.pt")


def run_5a(pilot: bool = False, resume_latest: bool = False, max_wall_hours: float = 0.0) -> None:
    """5A: Frozen backbone + v2 objective (primary redesign run)."""
    print("\n=== RUN 5A: Frozen backbone + v2 objective ===")
    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=20,
                              sigma_min=0.01, sigma_max=1.0, control_weight=1.0)
    cfg = ControlledScoreTrainingConfig(
        # Dataset
        dataset_name="fashionmnist",
        batch_size=32,
        epochs=20 if pilot else 100,
        dataset_subset_size=2000 if pilot else 0,
        # Architecture
        model=_SCORE_CFG,
        sde=sde_cfg,
        use_conv_control=True,
        conv_control=_CONV_CFG,
        # Pretrained score backbone (frozen)
        score_backbone_ckpt=_SCORE_CKPT,
        freeze_score_model=True,
        # v2 objective
        objective_version="v2",
        detach_control_energy=True,
        quality_weight=1.0,
        trajectory_quality_temperature=1.0,
        # Path KL: small weight, warmed up
        path_kl_weight=0.01,
        warmup_epochs=5,
        warmup_ramp_epochs=5,
        target_path_kl_weight=0.01,
        # REINFORCE: warmed up alongside path KL
        reinforce_weight=0.1,
        target_reinforce_weight=0.1,
        # LR: TwoPhaseScheduler
        use_two_phase_scheduler=True,
        use_lr_schedule=False,
        phase1_epochs=5,
        phase1_lr=1e-5,
        phase2_lr=5e-4,
        lr=5e-4,
        lr_min=5e-7,
        # Scaled random init (break zero attractor)
        scaled_output_init=True,
        output_init_std=1e-4,
        # Other
        control_weight=0.0,
        grad_clip=1.0,
        seed=42,
        ema_decay=0.999,
        # Checkpointing
        checkpoint_dir=str(ROOT / "checkpoints" / ("controlled_v2_5a_pilot" if pilot else "controlled_v2_5a_seed42")),
        save_every_n_epochs=5 if pilot else 10,
        log_dir=str(ROOT / "data" / "logs"),
        run_name="controlled_v2_5a_pilot" if pilot else "controlled_v2_5a_seed42",
        jsonl_log=str(ROOT / "data" / "logs" / ("controlled_v2_5a_pilot.jsonl" if pilot else "controlled_v2_5a_seed42.jsonl")),
        resume_latest=resume_latest,
        checkpoint_every_n_minutes=30,
        max_wall_hours=max_wall_hours,
    )
    result = train_controlled_score(cfg)
    print(f"\nRun 5A complete: {result}")


def run_5b(resume_latest: bool = False, max_wall_hours: float = 0.0) -> None:
    """5B: Joint fine-tuning (score + controller, distillation regularizer)."""
    print("\n=== RUN 5B: Joint fine-tune + distillation ===")
    cfg = ControlledScoreTrainingConfig(
        dataset_name="fashionmnist",
        batch_size=32,
        epochs=100,
        model=_SCORE_CFG,
        sde=_SDE_CFG,
        use_conv_control=True,
        conv_control=_CONV_CFG,
        score_backbone_ckpt=_SCORE_CKPT,
        freeze_score_model=False,  # joint fine-tune
        # v2 objective
        objective_version="v2",
        detach_control_energy=True,
        quality_weight=1.0,
        trajectory_quality_temperature=1.0,
        path_kl_weight=0.01,
        warmup_epochs=5,
        warmup_ramp_epochs=5,
        target_path_kl_weight=0.01,
        reinforce_weight=0.1,
        target_reinforce_weight=0.1,
        # Joint fine-tune: score model at 1% of controller LR
        score_lr_multiplier=0.01,
        # Distillation: keep score near pretrained reference
        distillation_weight=10.0,
        use_two_phase_scheduler=True,
        use_lr_schedule=False,
        phase1_epochs=5,
        phase1_lr=1e-5,
        phase2_lr=5e-4,
        lr=5e-4,
        lr_min=5e-7,
        scaled_output_init=True,
        output_init_std=1e-4,
        control_weight=0.0,
        grad_clip=1.0,
        seed=42,
        ema_decay=0.999,
        checkpoint_dir=str(ROOT / "checkpoints" / "controlled_v2_5b_joint"),
        save_every_n_epochs=10,
        log_dir=str(ROOT / "data" / "logs"),
        run_name="controlled_v2_5b_joint",
        jsonl_log=str(ROOT / "data" / "logs" / "controlled_v2_5b_joint.jsonl"),
        resume_latest=resume_latest,
        checkpoint_every_n_minutes=30,
        max_wall_hours=max_wall_hours,
    )
    result = train_controlled_score(cfg)
    print(f"\nRun 5B complete: {result}")


def run_5c(resume_latest: bool = False, max_wall_hours: float = 0.0) -> None:
    """5C: Path-KL ablation (path_kl_weight=0, quality signal only)."""
    print("\n=== RUN 5C: Path-KL ablation (no path KL) ===")
    cfg = ControlledScoreTrainingConfig(
        dataset_name="fashionmnist",
        batch_size=32,
        epochs=100,
        model=_SCORE_CFG,
        sde=_SDE_CFG,
        use_conv_control=True,
        conv_control=_CONV_CFG,
        score_backbone_ckpt=_SCORE_CKPT,
        freeze_score_model=True,
        # v2 objective without path KL
        objective_version="v2",
        detach_control_energy=True,
        quality_weight=1.0,
        trajectory_quality_temperature=1.0,
        path_kl_weight=0.0,      # ablated
        warmup_epochs=0,
        warmup_ramp_epochs=0,
        target_path_kl_weight=0.0,
        reinforce_weight=0.1,
        target_reinforce_weight=0.1,
        use_two_phase_scheduler=True,
        use_lr_schedule=False,
        phase1_epochs=5,
        phase1_lr=1e-5,
        phase2_lr=5e-4,
        lr=5e-4,
        lr_min=5e-7,
        scaled_output_init=True,
        output_init_std=1e-4,
        control_weight=0.0,
        grad_clip=1.0,
        seed=42,
        ema_decay=0.999,
        checkpoint_dir=str(ROOT / "checkpoints" / "controlled_v2_5c_nopkL"),
        save_every_n_epochs=10,
        log_dir=str(ROOT / "data" / "logs"),
        run_name="controlled_v2_5c_nopkL",
        jsonl_log=str(ROOT / "data" / "logs" / "controlled_v2_5c_nopkL.jsonl"),
        resume_latest=resume_latest,
        checkpoint_every_n_minutes=30,
        max_wall_hours=max_wall_hours,
    )
    result = train_controlled_score(cfg)
    print(f"\nRun 5C complete: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["5a", "5b", "5c", "all"], default="5a")
    parser.add_argument("--pilot", action="store_true",
                        help="Fast pilot: 2000 samples, 20 epochs (~30 min on GTX 1650 Ti)")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume from autosave.pt or latest checkpoint (used by segmented launcher).")
    parser.add_argument("--max-wall-hours", type=float, default=0.0,
                        help="Exit with code 42 after this many wall-clock hours (used by segmented launcher).")
    args = parser.parse_args()

    if args.run in ("5a", "all"):
        run_5a(pilot=args.pilot, resume_latest=args.resume_latest, max_wall_hours=args.max_wall_hours)
    if args.run in ("5b", "all"):
        run_5b(resume_latest=args.resume_latest, max_wall_hours=args.max_wall_hours)
    if args.run in ("5c", "all"):
        run_5c(resume_latest=args.resume_latest, max_wall_hours=args.max_wall_hours)


if __name__ == "__main__":
    main()
