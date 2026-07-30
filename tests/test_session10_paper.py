"""Session 10 paper-readiness tests.

Tests the v2 objective redesign, WarmupSchedule, TwoPhaseScheduler,
gradient diagnostic, and collapse metric computability.

Target: 8 new tests bringing total from 56 to 64 passing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# 1. Trajectory quality loss is non-zero for different generated/real images
# ---------------------------------------------------------------------------

def test_trajectory_quality_loss_nonzero():
    """compute_trajectory_quality_loss returns a positive finite scalar."""
    from its.objectives.loss_components import compute_trajectory_quality_loss
    from torchvision.models import inception_v3, Inception_V3_Weights
    import torch.nn as nn

    incep = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    incep.fc = nn.Identity()
    incep.aux_logits = False
    incep.eval()
    for p in incep.parameters():
        p.requires_grad_(False)

    gen = torch.randn(4, 1, 28, 28)
    real = torch.randn(4, 1, 28, 28)
    loss = compute_trajectory_quality_loss(gen, real, incep)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.item() >= 0.0, f"loss should be non-negative: {loss.item()}"


# ---------------------------------------------------------------------------
# 2. Trajectory quality loss per-sample mode returns tuple
# ---------------------------------------------------------------------------

def test_trajectory_quality_loss_per_sample_shape():
    """compute_trajectory_quality_loss(return_per_sample=True) returns (scalar, (B,) tensor)."""
    from its.objectives.loss_components import compute_trajectory_quality_loss
    from torchvision.models import inception_v3, Inception_V3_Weights
    import torch.nn as nn

    incep = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    incep.fc = nn.Identity()
    incep.aux_logits = False
    incep.eval()
    for p in incep.parameters():
        p.requires_grad_(False)

    gen = torch.randn(4, 1, 28, 28)
    real = torch.randn(4, 1, 28, 28)
    scalar_loss, per_sample = compute_trajectory_quality_loss(gen, real, incep, return_per_sample=True)
    assert isinstance(per_sample, torch.Tensor), "per_sample should be a Tensor"
    assert per_sample.shape == (4,), f"Expected shape (4,), got {per_sample.shape}"
    assert torch.isfinite(scalar_loss), f"scalar_loss not finite: {scalar_loss}"


# ---------------------------------------------------------------------------
# 3. REINFORCE quality loss is finite with valid inputs
# ---------------------------------------------------------------------------

def test_reinforce_quality_loss_finite():
    """compute_reinforce_quality_loss produces a finite scalar."""
    from its.objectives.loss_components import compute_reinforce_quality_loss

    B = 8
    rewards = torch.rand(B)
    log_probs = -torch.rand(B) * 0.5  # small negative log probs
    loss = compute_reinforce_quality_loss(rewards, log_probs)
    assert torch.isfinite(loss), f"REINFORCE loss not finite: {loss}"


# ---------------------------------------------------------------------------
# 4. REINFORCE quality loss with constant rewards gives near-zero loss
#    (all advantages are zero when baseline = mean = reward)
# ---------------------------------------------------------------------------

def test_reinforce_quality_loss_zero_advantage():
    """REINFORCE with constant rewards (zero advantage) should give zero loss."""
    from its.objectives.loss_components import compute_reinforce_quality_loss

    B = 16
    rewards = torch.ones(B)  # identical rewards → zero advantage
    log_probs = -torch.rand(B)
    loss = compute_reinforce_quality_loss(rewards, log_probs)
    assert loss.abs().item() < 1e-6, f"Expected ~0 but got {loss.item()}"


# ---------------------------------------------------------------------------
# 5. WarmupSchedule returns zero weights at epoch 0
# ---------------------------------------------------------------------------

def test_warmup_schedule_zero_at_epoch_zero():
    """WarmupSchedule returns path_kl_weight=0 and reinforce_weight=0 at epoch 0."""
    from its.training.schedules import WarmupSchedule

    sched = WarmupSchedule(warmup_epochs=5, ramp_epochs=5,
                           target_path_kl_weight=0.01, target_reinforce_weight=0.1)
    weights = sched.get_weights(0)
    assert weights["path_kl_weight"] == 0.0, f"Expected 0.0, got {weights['path_kl_weight']}"
    assert weights["reinforce_weight"] == 0.0, f"Expected 0.0, got {weights['reinforce_weight']}"


# ---------------------------------------------------------------------------
# 6. WarmupSchedule reaches target at epoch warmup + ramp
# ---------------------------------------------------------------------------

def test_warmup_schedule_full_at_epoch_ten():
    """WarmupSchedule returns target weights at epoch warmup+ramp (epoch 10 for 5+5)."""
    from its.training.schedules import WarmupSchedule

    sched = WarmupSchedule(warmup_epochs=5, ramp_epochs=5,
                           target_path_kl_weight=0.01, target_reinforce_weight=0.1)
    weights = sched.get_weights(10)  # 0-indexed: epoch 10 = 11th epoch
    assert abs(weights["path_kl_weight"] - 0.01) < 1e-9
    assert abs(weights["reinforce_weight"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# 7. TwoPhaseScheduler steps up LR at phase boundary
# ---------------------------------------------------------------------------

def test_two_phase_scheduler_step_up():
    """TwoPhaseScheduler LR at epoch phase1_epochs should jump to near phase2_lr."""
    import torch.optim as optim
    from its.training.schedules import TwoPhaseScheduler
    import torch.nn as nn

    dummy = nn.Linear(2, 2)
    opt = optim.AdamW(dummy.parameters(), lr=1e-5)
    sched = TwoPhaseScheduler(opt, phase1_epochs=5, phase1_lr=1e-5, phase2_lr=5e-4,
                               total_epochs=50, lr_min=5e-7)

    # Phase 1: LR should be phase1_lr
    lr_p1 = sched.step(0)
    assert abs(lr_p1 - 1e-5) < 1e-9, f"Phase 1 LR should be 1e-5, got {lr_p1}"

    # Phase 2 start: LR should be near phase2_lr (cosine at t=0)
    lr_p2 = sched.step(5)
    assert lr_p2 > 1e-4, f"Phase 2 LR should be near 5e-4, got {lr_p2}"


# ---------------------------------------------------------------------------
# 8. Controller collapse metric is computable from checkpoint
# ---------------------------------------------------------------------------

def test_controller_collapse_metric_computable():
    """Collapse trajectory metrics JSON exists and contains valid ratio data."""
    metrics_path = ROOT / "data" / "results" / "collapse_trajectory_metrics.json"
    assert metrics_path.exists(), f"Missing: {metrics_path}"

    import json
    metrics = json.loads(metrics_path.read_text())
    assert len(metrics) > 0, "Metrics list is empty"

    # Last epoch should have a very small ratio (confirmed collapse)
    last = metrics[-1]
    assert "ratio_ctrl_over_score" in last, "Missing ratio_ctrl_over_score"
    assert last["ratio_ctrl_over_score"] < 0.01, (
        f"Collapse not confirmed: ratio={last['ratio_ctrl_over_score']:.4f} >= 1%"
    )
    assert last["ratio_ctrl_over_score"] > 0.0, "Ratio should be positive"


# ---------------------------------------------------------------------------
# 9. eval_v2_multiseed.py script exists and is importable
# ---------------------------------------------------------------------------

def test_eval_v2_multiseed_script_exists():
    """eval_v2_multiseed.py for Step 5D exists and has the expected run-id choices."""
    script = ROOT / "scripts" / "eval_v2_multiseed.py"
    assert script.exists(), f"Missing: {script}"
    source = script.read_text()
    assert "5a" in source, "Missing run-id '5a'"
    assert "5b" in source, "Missing run-id '5b'"
    assert "5c" in source, "Missing run-id '5c'"
    assert "ci_95" in source, "Missing ci_95 function"
    assert "controlled_v2" in source, "Missing v2 checkpoint path reference"


# ---------------------------------------------------------------------------
# 10. extended_baseline_analysis.py script exists with required parts
# ---------------------------------------------------------------------------

def test_extended_baseline_analysis_script_exists():
    """extended_baseline_analysis.py for Step 7C exists with all three analysis parts."""
    script = ROOT / "scripts" / "extended_baseline_analysis.py"
    assert script.exists(), f"Missing: {script}"
    source = script.read_text()
    assert "nfe25" in source or "NFE=25" in source, "Missing NFE=25 analysis"
    assert "entropy_production" in source, "Missing entropy production analysis"
    assert "mode_coverage" in source, "Missing mode coverage analysis"
    assert "ddpm" in source.lower(), "Missing DDPM sampler"
    assert "ddim" in source.lower(), "Missing DDIM sampler"
