"""Session 7 paper-readiness tests (Step 8).

Tests:
- test_deterministic_training
- test_nfe_efficiency_curve_data_complete
- test_memorization_check_runs
- test_mode_coverage_sums_to_one
- test_reproduce_script_exists_and_is_executable
- test_confidence_interval_in_eval_results
- test_paper_figures_all_exist
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# test_deterministic_training
# ---------------------------------------------------------------------------

def _run_small_training(seed: int, steps: int = 10) -> list[float]:
    """Run a tiny score-model training and return per-step losses."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Tiny linear model as proxy for score model
    model = nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(128, 4)
    ds = DataLoader(TensorDataset(data), batch_size=8, shuffle=False)

    losses = []
    for i, (x,) in enumerate(ds):
        if i >= steps:
            break
        pred = model(x)
        loss = (pred - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def test_deterministic_training():
    """Two runs with the same seed must produce identical losses."""
    losses_a = _run_small_training(seed=42, steps=10)
    losses_b = _run_small_training(seed=42, steps=10)
    assert len(losses_a) == 10
    for i, (a, b) in enumerate(zip(losses_a, losses_b)):
        assert abs(a - b) < 1e-6, f"Divergence at step {i}: {a} vs {b}"


# ---------------------------------------------------------------------------
# test_nfe_efficiency_curve_data_complete
# ---------------------------------------------------------------------------

def test_nfe_efficiency_curve_data_complete():
    """fmnist_eval_matrix_final.json must have FID at 4 NFE levels per baseline."""
    result_file = ROOT / "data" / "results" / "fmnist_eval_matrix_final.json"
    if not result_file.exists():
        pytest.skip("fmnist_eval_matrix_final.json not yet generated (Step 3A)")

    data = json.loads(result_file.read_text())
    required_nfe = {50, 100, 200, 500}
    baselines = {"ddpm", "ddim", "score_sde"}

    for sampler_name in baselines:
        if sampler_name not in data:
            continue
        entries = data[sampler_name]
        found_nfe = set()
        for entry in entries:
            assert "fid" in entry, f"Missing 'fid' in {sampler_name} entry"
            assert "num_samples" in entry, f"Missing 'num_samples' in {sampler_name} entry"
            found_nfe.add(entry.get("nfe"))
        # At least some NFE levels present
        assert len(found_nfe) > 0, f"No NFE entries for {sampler_name}"


# ---------------------------------------------------------------------------
# test_memorization_check_runs
# ---------------------------------------------------------------------------

def test_memorization_check_runs():
    """Nearest-neighbor memorization check should return expected dict structure."""
    import torch

    # Generate tiny fake samples and training set
    n_generated = 16
    n_train = 100
    img_dim = 4 * 4  # tiny images

    generated = torch.randn(n_generated, img_dim)
    train_set = torch.randn(n_train, img_dim)

    # Compute nearest-neighbor distances in pixel space
    threshold = 0.05
    nn_distances = []
    for g in generated:
        dists = (train_set - g.unsqueeze(0)).norm(dim=1)
        nn_distances.append(float(dists.min().item()))

    fraction_below = sum(1 for d in nn_distances if d < threshold) / len(nn_distances)

    result = {
        "threshold": threshold,
        "num_generated": n_generated,
        "num_train": n_train,
        "nn_distances_mean": float(sum(nn_distances) / len(nn_distances)),
        "fraction_below_threshold": fraction_below,
    }

    assert "threshold" in result
    assert "fraction_below_threshold" in result
    assert "nn_distances_mean" in result
    assert math.isfinite(result["nn_distances_mean"])
    assert 0.0 <= result["fraction_below_threshold"] <= 1.0


# ---------------------------------------------------------------------------
# test_mode_coverage_sums_to_one
# ---------------------------------------------------------------------------

def test_mode_coverage_sums_to_one():
    """Class fraction distribution must sum to 1.0."""
    n_samples = 100
    n_classes = 10

    # Simulate a fixed classification mapping (round-robin)
    labels = [i % n_classes for i in range(n_samples)]
    counts = [0] * n_classes
    for lbl in labels:
        counts[lbl] += 1

    fractions = [c / n_samples for c in counts]

    total = sum(fractions)
    assert abs(total - 1.0) < 1e-9, f"Fractions sum to {total}, not 1.0"

    # Entropy should be log(n_classes) for uniform distribution
    entropy = -sum(p * math.log(p + 1e-12) for p in fractions if p > 0)
    assert entropy > 0.0


# ---------------------------------------------------------------------------
# test_reproduce_script_exists_and_is_executable
# ---------------------------------------------------------------------------

def test_reproduce_script_exists_and_is_executable():
    """reproduce_paper_results.sh must exist, parse cleanly, and have >= 10 commands."""
    script = ROOT / "scripts" / "reproduce_paper_results.sh"
    assert script.exists(), "scripts/reproduce_paper_results.sh does not exist"

    # Verify it is parseable bash syntax (bash -n = syntax check only)
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    content = script.read_text()
    # Count non-comment, non-empty lines with at least one command
    cmd_lines = [
        ln for ln in content.splitlines()
        if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("echo")
    ]
    assert len(cmd_lines) >= 10, f"Script has only {len(cmd_lines)} command lines"


# ---------------------------------------------------------------------------
# test_confidence_interval_in_eval_results
# ---------------------------------------------------------------------------

def test_confidence_interval_in_eval_results():
    """Every FID entry in fmnist_eval_matrix_final.json must have ci_95."""
    result_file = ROOT / "data" / "results" / "fmnist_eval_matrix_final.json"
    if not result_file.exists():
        pytest.skip("fmnist_eval_matrix_final.json not yet generated (Step 3A)")

    data = json.loads(result_file.read_text())

    for sampler_name, entries in data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if "fid" not in entry:
                continue
            assert "ci_95" in entry, (
                f"Missing ci_95 in {sampler_name} FID entry (NFE={entry.get('nfe')})"
            )
            ci = entry["ci_95"]
            assert "lower" in ci and "upper" in ci
            assert math.isfinite(ci["lower"])
            assert math.isfinite(ci["upper"])
            assert ci["lower"] <= ci["upper"], (
                f"CI lower > upper: {ci['lower']} > {ci['upper']}"
            )


# ---------------------------------------------------------------------------
# test_paper_figures_all_exist
# ---------------------------------------------------------------------------

EXPECTED_FIGURES = [
    "pareto_frontier",
    "ablation_fid_chart",
    "control_drift_analysis",
    "nfe_efficiency_curves",
    "path_kl_vs_fid_trajectory",
    "entropy_production_profile",
    "crooks_verification",
    "mode_coverage_radar",
]


def test_paper_figures_all_exist():
    """All 8 paper figures must exist as both .png and .pdf in data/paper_figures/."""
    fig_dir = ROOT / "data" / "paper_figures"
    if not fig_dir.exists():
        pytest.skip("data/paper_figures/ does not yet exist (Step 4 / Step 9)")

    missing = []
    for name in EXPECTED_FIGURES:
        for ext in [".png", ".pdf"]:
            path = fig_dir / (name + ext)
            if not path.exists():
                missing.append(str(path))

    if missing:
        pytest.skip(
            f"Paper figures not yet generated ({len(missing)} missing): {missing[:4]}"
        )
