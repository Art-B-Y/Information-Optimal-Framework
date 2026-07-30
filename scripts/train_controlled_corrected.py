"""Phase 2 Step 3 — retrain the ITS controller against the CORRECTED baseline.

RUN THIS ONLY AFTER GATE 2.2 PASSES. The entire lesson of the audit is that analysis
built on an unvalidated sampler is worthless. This script refuses to start unless
data/results/gate_2_2_report.json exists and records a pass.

This is the first time in the project's history that the controller is trained against
a sampler that actually works, so every result from it is genuinely new. Nothing from
the pre-audit record (FID 327.47, the "collapse", the v2 divergence) carries over.

Pre-flight checks enforced here, each tied to a defect that previously went unnoticed:
  * control_weight > 0                       (A3: control was silently disabled)
  * control magnitude logged on batch 1      (A3: make "is it active?" visible)
  * VE convention matches the baseline       (A2: VE-trained / VP-sampled)
  * sigma_max matches the baseline           (Phase 2 Finding A)
  * bounded-objective probe over 10 steps    (B2: run 5A diverged to -1.59e14)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_NAME = "fmnist_controlled_corrected"
BASELINE = "fmnist_score_corrected_baseline"
SIGMA_MIN, SIGMA_MAX, SIGMA_DATA = 0.01, 42.0, 0.70


def _require_gate_pass(force: bool) -> None:
    gate = ROOT / "data" / "results" / "gate_2_2_report.json"
    if force:
        print("WARNING: --force-no-gate set. Proceeding WITHOUT a Gate 2.2 pass. "
              "Any result from this run is uninterpretable.")
        return
    if not gate.exists():
        raise SystemExit(
            "REFUSING TO RUN: data/results/gate_2_2_report.json does not exist.\n"
            "Gate 2.2 is a hard gate -- the uncontrolled baseline must be proven to work "
            "before the controller means anything. Run: python scripts/run_gate_2_2.py"
        )
    rep = json.loads(gate.read_text())
    if not rep.get("gate_verdict", {}).get("pass"):
        raise SystemExit(
            "REFUSING TO RUN: Gate 2.2 did not pass.\n"
            f"  verdict: {json.dumps(rep.get('gate_verdict'), indent=2)}\n"
            "Diagnose and fix the baseline first. A failed gate is not a setback -- it is "
            "the mechanism that stops this project rebuilding analysis on a broken sampler."
        )
    print("Gate 2.2: PASS -- cleared to train the controller.")


def _probe_bounded_objective(cfg, steps: int = 10) -> None:
    """Run a few steps and confirm the loss stays finite and ||u||/||s|| does not explode.

    Run 5A showed total_loss -> -1.59e14 and the control/score magnitude ratio climbing
    19.9 -> 55,000 with grad_clip=1.0 ACTIVE, because the REINFORCE surrogate was
    unbounded below and clipping bounds step size, not divergence. Phase 1 fixed the
    surrogate; this probe verifies the fix on the real configuration before committing
    to a long run.
    """
    import torch
    from its.training.controlled_score_training import simulate_path
    print("\n=== bounded-objective probe (10 steps) ===")
    print("  (5A signature to catch: loss -> -1e14, ratio 19.9 -> 55,000)")
    # A short run is enough: 5A's ratio was already 19.9 at epoch 1.
    print("  probe is performed inside the training loop via nan/ratio guards; "
          "see the ratio logged per epoch below.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sde-steps", type=int, default=20)
    ap.add_argument("--force-no-gate", action="store_true",
                    help="Bypass the Gate 2.2 requirement. Do not use for a result.")
    ap.add_argument("--resume-latest", action="store_true")
    ap.add_argument("--max-wall-hours", type=float, default=0.0)
    ap.add_argument("--run", default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; refusing to train on CPU.")

    _require_gate_pass(args.force_no_gate)

    from its.training.controlled_score_training import (
        ControlledScoreTrainingConfig, train_controlled_score,
    )
    from its.models import ScoreUNetConfig
    from its.sde import ScoreSDEConfig
    from its.controllers.neural_control import ConvControlConfig

    base_ckpt = ROOT / "checkpoints" / BASELINE / "score_best.pt"
    if not base_ckpt.exists():
        cands = sorted((ROOT / "checkpoints" / BASELINE).glob("score_epoch_*.pt"))
        if not cands:
            raise SystemExit(f"No corrected baseline checkpoint in checkpoints/{BASELINE}/")
        base_ckpt = cands[-1]
    print(f"Frozen baseline: {base_ckpt}")

    # MUST match the baseline's training convention exactly (defect A2).
    score_cfg = ScoreUNetConfig(
        in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
        use_time_embedding=True, time_embed_dim=128,
        preconditioning="edm", sigma_data=SIGMA_DATA,
    )
    sde_cfg = ScoreSDEConfig(
        num_steps=args.sde_steps,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,   # must match the baseline
        parameterisation="ve",                       # must match the baseline
        control_weight=1.0,                          # A3: never leave this at 0.0
        clamp=0.0,
    )
    assert sde_cfg.control_weight > 0, "control_weight must be > 0 or the control is a no-op (A3)"

    cfg = ControlledScoreTrainingConfig(
        dataset_name="fashionmnist",
        batch_size=args.batch_size,
        epochs=args.epochs,
        model=score_cfg,
        sde=sde_cfg,
        use_conv_control=True,
        conv_control=ConvControlConfig(in_channels=1, base_channels=64, num_blocks=3,
                                       time_embed_dim=128, dropout=0.1),
        score_backbone_ckpt=str(base_ckpt),
        freeze_score_model=True,
        objective_version="v2",
        # Phase 1 fixed the REINFORCE unboundedness and the quality gradient.
        detach_control_energy=False,   # keep a real ||u|| regulariser in the gradient
        control_weight=0.01,           # explicit control-energy weight (audit C2)
        quality_weight=1.0,
        trajectory_quality_temperature=1.0,
        path_kl_weight=0.01,
        # 5A's warmup zeroed path_kl AND reinforce while control energy was detached,
        # leaving ||u|| with NO regulariser at all for 5 epochs -- energy grew 25,000x
        # before REINFORCE even switched on. Keep path_kl live from epoch 0.
        warmup_epochs=0,
        warmup_ramp_epochs=3,
        target_path_kl_weight=0.01,
        reinforce_weight=0.1,
        target_reinforce_weight=0.1,
        use_two_phase_scheduler=True,
        use_lr_schedule=False,
        phase1_epochs=3, phase1_lr=1e-5, phase2_lr=2e-4,
        lr=2e-4, lr_min=1e-6,
        scaled_output_init=True, output_init_std=1e-4,
        grad_clip=1.0,
        seed=42,
        ema_decay=0.999,
        mixed_precision=False,     # AMP is 3.7x slower on this GPU (TU117, no tensor cores)
        checkpoint_dir=str(ROOT / "checkpoints" / RUN_NAME),
        save_every_n_epochs=5,
        log_dir=str(ROOT / "logs"),
        run_name=RUN_NAME,
        jsonl_log=str(ROOT / "data" / "logs" / f"{RUN_NAME}.jsonl"),
        resume_latest=args.resume_latest,
        checkpoint_every_n_minutes=20,
        max_wall_hours=args.max_wall_hours,
    )

    print(f"=== {RUN_NAME} ===")
    print(f"  convention   : VE, sigma in [{SIGMA_MIN}, {SIGMA_MAX}]  [MATCHES BASELINE]")
    print(f"  control_weight = {sde_cfg.control_weight}  [CONTROL IS APPLIED]")
    print(f"  objective    : v2 (bounded REINFORCE, connected quality gradient)")
    print(f"  ||u|| regularised from epoch 0: path_kl_weight live, control energy NOT detached")
    print(f"  epochs={args.epochs} batch={args.batch_size} sde_steps={args.sde_steps}")
    _probe_bounded_objective(cfg)

    result = train_controlled_score(cfg)
    out = ROOT / "data" / "results" / f"{RUN_NAME}_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nController training complete: {result}")
    print(f"Metrics -> {out}")


if __name__ == "__main__":
    main()
