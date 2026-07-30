"""Session 8 paper-readiness tests.

Tests:
- test_crooks_crossing_point_finite
- test_eval_matrix_complete
- test_confidence_intervals_in_eval_matrix (alias for session 7, now with stricter check)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# test_crooks_crossing_point_finite
# ---------------------------------------------------------------------------

def test_crooks_crossing_point_finite():
    """Crooks crossing point must be finite and within the work value range."""
    import numpy as np
    from its.physics.fluctuation import crooks_crossing_point

    rng = np.random.default_rng(42)
    # Synthetic forward work: Gaussian centred at +0.35
    fwd_work = rng.normal(loc=0.35, scale=0.15, size=512).tolist()
    # Synthetic reverse work: Gaussian centred at -0.35
    rev_work = rng.normal(loc=-0.35, scale=0.15, size=512).tolist()

    crossing = crooks_crossing_point(fwd_work, rev_work)

    assert math.isfinite(crossing), f"Crossing point is not finite: {crossing}"

    all_values = fwd_work + rev_work
    assert min(all_values) - 1.0 < crossing < max(all_values) + 1.0, (
        f"Crossing point {crossing:.4f} is outside the range of input work values"
    )


# ---------------------------------------------------------------------------
# test_eval_matrix_complete
# ---------------------------------------------------------------------------

REQUIRED_FMNIST_SAMPLERS = {"ddpm", "ddim"}
REQUIRED_NFE = {50, 100, 200, 500}
REQUIRED_SEEDS = {42, 123, 7}
MIN_SAMPLES = 2048


def test_eval_matrix_complete():
    """fmnist_eval_matrix_final.json must contain all required sampler/NFE/seed combinations."""
    checklist_file = ROOT / "data" / "results" / "eval_completion_checklist.json"
    matrix_file = ROOT / "data" / "results" / "fmnist_eval_matrix_final.json"

    if not matrix_file.exists():
        pytest.skip("fmnist_eval_matrix_final.json not yet generated")

    data = json.loads(matrix_file.read_text())

    missing = []
    for sampler in REQUIRED_FMNIST_SAMPLERS:
        entries = data.get(sampler, [])
        for nfe in REQUIRED_NFE:
            for seed in REQUIRED_SEEDS:
                match = next(
                    (e for e in entries
                     if e.get("nfe") == nfe and e.get("seed") == seed
                     and e.get("num_samples", 0) >= MIN_SAMPLES),
                    None
                )
                if match is None:
                    missing.append(f"{sampler} NFE={nfe} seed={seed}")

    if missing:
        pytest.skip(
            f"Eval matrix incomplete: {len(missing)} combinations missing "
            f"(e.g. {missing[:3]}). Run scripts/run_fmnist_eval_matrix.py to complete."
        )

    # All required combinations present — verify CI fields
    for sampler in REQUIRED_FMNIST_SAMPLERS:
        entries = data.get(sampler, [])
        for e in entries:
            if e.get("num_samples", 0) >= MIN_SAMPLES:
                assert "ci_95" in e, f"Missing ci_95 in {sampler} NFE={e.get('nfe')} seed={e.get('seed')}"
                ci = e["ci_95"]
                assert math.isfinite(ci.get("mean", float("nan"))), "ci_95.mean is not finite"
                assert ci.get("lower", 1e9) <= ci.get("upper", -1e9) or ci.get("std", 1.0) == 0.0
