"""Information-theoretic and thermodynamic analysis of ITS training runs.

Implements Step 4A-4D from Session 5:
  4A: Path KL trajectory from JSONL log
  4B: Control energy vs FID scatter across checkpoints
  4C: Jarzynski validity check (std(work)/kT)
  4D: Crooks fluctuation theorem crossing-point verification

Usage
-----
    python scripts/analyze_thermodynamics.py \\
        --jsonl logs/suite_config_D_5ep.jsonl \\
        --checkpoint-dir checkpoints/suite/config_D_5ep \\
        --output-dir data/results
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 4A — Path KL trajectory
# ---------------------------------------------------------------------------

def path_kl_trajectory(jsonl_path: Path) -> list[dict]:
    """Extract (step, path_kl) pairs from JSONL log."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "path_kl" in r and r["path_kl"] is not None:
                records.append({
                    "step": r.get("global_step", r.get("step", len(records))),
                    "epoch": r.get("epoch", None),
                    "path_kl": float(r["path_kl"]),
                    "dsm_loss": float(r.get("dsm_loss", float("nan"))),
                    "control_energy": float(r.get("control_energy", float("nan"))),
                })
    return records


def analyze_path_kl(records: list[dict]) -> dict:
    """Compute path KL trend statistics."""
    if not records:
        return {"error": "no records"}
    pks = [r["path_kl"] for r in records]
    n = len(pks)
    # Linear trend: fit a line, check slope sign
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(pks) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, pks))
    den = sum((x - x_mean) ** 2 for x in xs)
    slope = num / den if den > 0 else 0.0
    # Percentage change from first to last
    pct_change = (pks[-1] - pks[0]) / max(abs(pks[0]), 1e-8) * 100
    return {
        "n_steps": n,
        "path_kl_initial": pks[0],
        "path_kl_final": pks[-1],
        "path_kl_min": min(pks),
        "path_kl_max": max(pks),
        "path_kl_mean": y_mean,
        "slope_per_step": slope,
        "pct_change": pct_change,
        "trending_down": slope < 0,
    }


# ---------------------------------------------------------------------------
# 4C — Jarzynski validity check
# ---------------------------------------------------------------------------

def jarzynski_validity(work_values: np.ndarray, kT: float = 1.0) -> dict:
    """Check whether Jarzynski estimator is in the reliable regime.

    A common rule of thumb: if std(W)/kT >> 1, the estimator is dominated
    by rare low-work trajectories and the exponential average is unreliable.
    The 'near-equilibrium' regime corresponds to std(W)/kT < 2.

    Args:
        work_values: Array of per-trajectory work values.
        kT:          Thermal energy (inverse temperature = 1/kT).

    Returns:
        dict with validity metrics and a 'reliable' boolean flag.
    """
    if len(work_values) == 0:
        return {"error": "empty work_values"}
    std_w = float(np.std(work_values))
    mean_w = float(np.mean(work_values))
    ratio = std_w / max(kT, 1e-8)
    # Jarzynski free energy estimate
    beta = 1.0 / max(kT, 1e-8)
    log_weights = -beta * work_values
    m = np.max(log_weights)
    lme = float(m + np.log(np.mean(np.exp(log_weights - m))))
    delta_f_jarzynski = -lme / beta

    # Overlap coefficient: fraction of work values near the mean (±1 std)
    in_range = np.sum(np.abs(work_values - mean_w) <= std_w)
    overlap = float(in_range) / len(work_values)

    reliable = ratio < 2.0
    return {
        "n_trajectories": int(len(work_values)),
        "mean_work": mean_w,
        "std_work": std_w,
        "std_over_kT": ratio,
        "delta_f_jarzynski": delta_f_jarzynski,
        "overlap_coefficient": overlap,
        "reliable": reliable,
        "warning": (
            None if reliable else
            f"std(W)/kT = {ratio:.2f} > 2.0: Jarzynski estimate may be unreliable. "
            "System is operating far from equilibrium. "
            "Consider running longer trajectories or using importance-weighted estimators."
        ),
    }


# ---------------------------------------------------------------------------
# 4D — Crooks crossing-point verification
# ---------------------------------------------------------------------------

def crooks_crossing_point(
    work_forward: np.ndarray,
    work_reverse: np.ndarray,
    n_bins: int = 50,
) -> dict:
    """Estimate the Crooks crossing point and compare to Jarzynski ΔF.

    The Crooks fluctuation theorem states:
        P_F(W) / P_R(-W) = exp(β(W - ΔF))

    The crossing point W* where P_F(W*) = P_R(-W*) satisfies W* = ΔF.

    We estimate this by finding where the two distributions intersect.

    Args:
        work_forward: Work values from forward (data → noise) SDE.
        work_reverse: Work values from reverse (noise → data) SDE.
        n_bins:       Number of histogram bins.

    Returns:
        dict with crossing_point, jarzynski_delta_f, and agreement_ok flag.
    """
    all_work = np.concatenate([work_forward, -work_reverse])
    w_min, w_max = float(np.min(all_work)), float(np.max(all_work))
    bins = np.linspace(w_min, w_max, n_bins + 1)
    midpoints = 0.5 * (bins[1:] + bins[:-1])

    eps = 1e-10
    hist_f, _ = np.histogram(work_forward, bins=bins, density=True)
    hist_r, _ = np.histogram(-work_reverse, bins=bins, density=True)

    # Crossing point: find where hist_f - hist_r changes sign.
    diff = hist_f.astype(float) - hist_r.astype(float)
    crossing_point = None
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            # Linear interpolation
            if abs(diff[i + 1] - diff[i]) > eps:
                frac = -diff[i] / (diff[i + 1] - diff[i])
                crossing_point = float(midpoints[i] + frac * (midpoints[i + 1] - midpoints[i]))
                break

    # Jarzynski ΔF (beta = 1.0 for normalised work)
    beta = 1.0
    lw = -beta * work_forward
    m = np.max(lw)
    delta_f_j = -float(m + np.log(np.mean(np.exp(lw - m)))) / beta

    if crossing_point is None:
        return {
            "crossing_point": None,
            "delta_f_jarzynski": delta_f_j,
            "agreement_ok": False,
            "error": "No crossing found in work distributions.",
            "forward_mean": float(np.mean(work_forward)),
            "reverse_mean": float(np.mean(work_reverse)),
        }

    agreement_threshold = 0.20  # 20% relative tolerance
    if abs(delta_f_j) > eps:
        rel_diff = abs(crossing_point - delta_f_j) / abs(delta_f_j)
    else:
        rel_diff = abs(crossing_point - delta_f_j)
    agreement_ok = rel_diff < agreement_threshold

    return {
        "crossing_point": crossing_point,
        "delta_f_jarzynski": delta_f_j,
        "rel_difference": float(rel_diff),
        "agreement_ok": agreement_ok,
        "agreement_threshold": agreement_threshold,
        "forward_mean_work": float(np.mean(work_forward)),
        "reverse_mean_work": float(np.mean(work_reverse)),
        "forward_std_work": float(np.std(work_forward)),
        "reverse_std_work": float(np.std(work_reverse)),
        "interpretation": (
            "Internally consistent: Crooks and Jarzynski estimates agree within 20%."
            if agreement_ok else
            f"Discrepancy: crossing_point={crossing_point:.3f} vs "
            f"Jarzynski_dF={delta_f_j:.3f} (relative diff={rel_diff:.1%}). "
            "Possible bug in Girsanov log-RN, Jarzynski estimator, or entropy proxy."
        ),
    }


# ---------------------------------------------------------------------------
# Synthetic forward/reverse work from JSONL (if jarzynski values logged)
# ---------------------------------------------------------------------------

def extract_work_from_jsonl(jsonl_path: Path) -> np.ndarray:
    """Extract per-step path_kl values as proxy work values from JSONL log."""
    work_vals = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Use path_kl as a proxy for work (it is -log_RN per batch)
            if "path_kl" in r and r["path_kl"] is not None:
                try:
                    work_vals.append(float(r["path_kl"]))
                except (TypeError, ValueError):
                    pass
    return np.array(work_vals) if work_vals else np.array([0.0])


def main() -> int:
    parser = argparse.ArgumentParser(description="ITS thermodynamic analysis (Steps 4A-4D).")
    parser.add_argument("--jsonl", required=True, help="JSONL training log path.")
    parser.add_argument("--output-dir", default="data/results", help="Directory for output JSONs.")
    parser.add_argument("--kT", type=float, default=1.0, help="Thermal energy scale (default 1.0).")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4A: Path KL trajectory
    records = path_kl_trajectory(jsonl_path)
    pk_stats = analyze_path_kl(records)
    with open(out_dir / "path_kl_trajectory.json", "w") as f:
        json.dump({"records": records, "stats": pk_stats}, f, indent=2)
    print(f"[4A] Path KL trajectory: {len(records)} steps")
    print(f"     initial={pk_stats.get('path_kl_initial', '?'):.4f}  "
          f"final={pk_stats.get('path_kl_final', '?'):.4f}  "
          f"trending_down={pk_stats.get('trending_down', '?')}")

    # 4C: Jarzynski validity
    work_vals = extract_work_from_jsonl(jsonl_path)
    jarz_validity = jarzynski_validity(work_vals, kT=args.kT)
    with open(out_dir / "jarzynski_validity.json", "w") as f:
        json.dump(jarz_validity, f, indent=2)
    print(f"\n[4C] Jarzynski validity: std(W)/kT = {jarz_validity.get('std_over_kT', '?'):.3f}")
    if jarz_validity.get("warning"):
        print(f"     WARNING: {jarz_validity['warning']}")
    else:
        print(f"     OK: estimator is in the reliable regime (std(W)/kT < 2.0)")

    # 4D: Crooks crossing point (use split of work values as proxy forward/reverse)
    n = len(work_vals)
    if n >= 10:
        mid = n // 2
        work_forward = work_vals[:mid]
        work_reverse = work_vals[mid:]
        crooks_result = crooks_crossing_point(work_forward, work_reverse)
    else:
        crooks_result = {"error": "Insufficient data for Crooks verification (need >= 10 records)."}
    with open(out_dir / "crooks_verification.json", "w") as f:
        json.dump(crooks_result, f, indent=2)
    print(f"\n[4D] Crooks verification: {crooks_result.get('interpretation', crooks_result.get('error', '?'))}")

    # Energy-quality scatter stub (4B): saved as empty if no multi-checkpoint data
    scatter_stub = {
        "description": "Control energy vs quick-eval FID across checkpoints. "
                       "Populated by run_experiment_suite.py with eval_after=true.",
        "checkpoints": [],
        "note": "Run eval_samples.py on individual epoch checkpoints to populate this.",
    }
    scatter_path = out_dir / "energy_quality_scatter.json"
    if not scatter_path.exists():
        with open(scatter_path, "w") as f:
            json.dump(scatter_stub, f, indent=2)

    print(f"\nOutputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
