"""Crooks fluctuation theorem verification.

Step 6 (Session 6). Verifies Crooks theorem holds for the controlled SDE:
the ratio P(W) / P(-W_reverse) = exp(beta * (W - DeltaF)) should be 0
(in log space, the signed area should be near 0).

Uses a synthetic Gaussian process for verification (no trained model needed).
Saves results to data/results/crooks_verification.json.

Usage:
    python scripts/run_crooks_verification.py
    python scripts/run_crooks_verification.py --n-trajectories 1000 --beta 1.0
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
    parser.add_argument("--n-trajectories", type=int, default=5000)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    import numpy as np
    from its.physics.fluctuation import jarzynski_work_estimate, crooks_ratio

    rng = np.random.default_rng(42)
    N = args.n_trajectories
    beta = args.beta
    dt = args.dt
    steps = args.steps

    print(f"Crooks verification: N={N}, beta={beta}, steps={steps}")

    # Synthetic protocol: harmonic trap shifted from k=1 to k=2
    # W = integral of dH/dlambda * dlambda over t in [0,1]
    # For a linear protocol lambda(t)=t: dH/dlambda = 0.5*x^2
    # True DeltaF = 0.5 * log(2) for harmonic trap k1=1 -> k2=2

    # Forward trajectories: start at N(0, 1), protocol shifts spring constant
    # Work = sum_t [ H(x_t, lambda_{t+1}) - H(x_t, lambda_t) ] * 1
    def _simulate_forward(n: int) -> np.ndarray:
        x = rng.standard_normal((n,))
        work = np.zeros(n)
        for step in range(steps):
            lam = step / steps       # lambda at current step
            lam_next = (step + 1) / steps
            # Hamiltonian H = 0.5 * (1 + lambda) * x^2
            # dW = H(x, lam_next) - H(x, lam) = 0.5 * (lam_next - lam) * x^2
            work += 0.5 * (lam_next - lam) * x ** 2
            # Langevin update: dx = -k*x*dt + sqrt(2/beta)*dW
            k = 1.0 + lam
            x += -k * x * dt + np.sqrt(2.0 / beta * dt) * rng.standard_normal(n)
        return work

    def _simulate_reverse(n: int) -> np.ndarray:
        # Reverse: start from equilibrium at k=2, reverse protocol lam: 1 -> 0
        x = rng.standard_normal((n,)) / np.sqrt(2.0)  # equilibrium at k=2
        work = np.zeros(n)
        for step in range(steps):
            lam = 1.0 - step / steps        # starts at 1, ends at 0
            lam_next = 1.0 - (step + 1) / steps
            work += 0.5 * (lam_next - lam) * x ** 2
            k = 1.0 + lam
            x += -k * x * dt + np.sqrt(2.0 / beta * dt) * rng.standard_normal(n)
        return work

    W_fwd = _simulate_forward(N)
    W_rev = _simulate_reverse(N)

    # Jarzynski estimate of DeltaF
    delta_f_fwd = jarzynski_work_estimate(W_fwd, beta)
    delta_f_rev = jarzynski_work_estimate(-W_rev, beta)  # sign flip for reverse

    # True DeltaF for harmonic k1=1, k2=2:
    # F = -log(Z), Z = sqrt(2*pi/k), so DeltaF = 0.5*log(k2/k1) = 0.5*log(2)
    import math
    true_delta_f = 0.5 * math.log(2.0)

    # Crooks crossing point: use a combined bin range covering both distributions.
    # Bins over the OVERLAP region of W_fwd and -W_rev.
    all_w = np.concatenate([W_fwd, -W_rev])
    w_lo, w_hi = np.percentile(all_w, 2), np.percentile(all_w, 98)
    bins = np.linspace(w_lo, w_hi, 80)
    hist_f, _ = np.histogram(W_fwd, bins=bins, density=True)
    hist_r, _ = np.histogram(-W_rev, bins=bins, density=True)
    midpoints = 0.5 * (bins[1:] + bins[:-1])

    # Check Crooks: log(P_F/P_R) should be linear in W with slope beta.
    # Fit a line to log(ratio) vs midpoints where both histograms > 1e-5.
    mask = (hist_f > 1e-5) & (hist_r > 1e-5)
    if mask.sum() >= 3:
        log_ratio = np.log((hist_f[mask] + 1e-10) / (hist_r[mask] + 1e-10))
        coeffs = np.polyfit(midpoints[mask], log_ratio, deg=1)
        crooks_slope = float(coeffs[0])   # should be near beta
        crooks_intercept = float(coeffs[1])   # should be near -beta * DeltaF
        crooks_delta_f_from_slope = -crooks_intercept / max(abs(crooks_slope), 1e-8)
        # Crossing: where log(ratio)=0, i.e. W = -intercept/slope
        if abs(crooks_slope) > 1e-8:
            crooks_crossing = float(-crooks_intercept / crooks_slope)
        else:
            crooks_crossing = float(np.nan)
    else:
        crooks_slope, crooks_intercept, crooks_delta_f_from_slope = 0.0, 0.0, 0.0
        crooks_crossing = float(np.nan)

    print(f"  True DeltaF:               {true_delta_f:.4f}")
    print(f"  Jarzynski (fwd):           {delta_f_fwd:.4f}")
    print(f"  Jarzynski (rev):           {delta_f_rev:.4f}")
    print(f"  Crooks crossing point:     {crooks_crossing:.4f} (should be near {true_delta_f:.4f})")
    print(f"  Crooks log-ratio slope:    {crooks_slope:.4f} (should be near beta={beta:.2f})")
    print(f"  Crooks-derived DeltaF:     {crooks_delta_f_from_slope:.4f}")

    # Verification criteria
    fwd_error = abs(delta_f_fwd - true_delta_f)
    rev_error = abs(delta_f_rev - true_delta_f)
    slope_error = abs(crooks_slope - beta)
    crossing_error = abs(crooks_crossing - true_delta_f)
    jarzynski_passed = fwd_error < 0.1  # tight tolerance for well-sampled estimate
    crooks_passed = slope_error < 0.5 and crossing_error < 0.3

    result = {
        "n_trajectories": N,
        "beta": beta,
        "steps": steps,
        "true_delta_f": true_delta_f,
        "jarzynski_fwd": delta_f_fwd,
        "jarzynski_rev": delta_f_rev,
        "jarzynski_fwd_error": fwd_error,
        "jarzynski_rev_error": rev_error,
        "crooks_crossing": crooks_crossing,
        "crooks_crossing_error": crossing_error,
        "crooks_log_ratio_slope": crooks_slope,
        "crooks_slope_error": slope_error,
        "crooks_delta_f": crooks_delta_f_from_slope,
        "jarzynski_passed": jarzynski_passed,
        "crooks_passed": crooks_passed,
        "overall_passed": jarzynski_passed and crooks_passed,
        # Keys for generate_paper_figures.py compatibility
        "forward_mean_work": float(W_fwd.mean()),
        "forward_std_work": float(W_fwd.std()),
        "reverse_mean_work": float(W_rev.mean()),
        "reverse_std_work": float(W_rev.std()),
        "crossing_point": crooks_crossing,
        "delta_f_jarzynski": delta_f_fwd,
    }

    out_path = ROOT / "data" / "results" / "crooks_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nJarzynski passed: {jarzynski_passed} (error={fwd_error:.4f})")
    print(f"Crooks passed: {crooks_passed} (slope_err={slope_error:.4f}, crossing_err={crossing_error:.4f})")
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
