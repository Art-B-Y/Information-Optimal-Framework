from __future__ import annotations

import math
import numpy as np
from typing import Optional


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-mean-exp: log(mean(exp(x)))."""
    m = np.max(x)
    return float(m + np.log(np.mean(np.exp(x - m))))


def jarzynski_work_estimate(
    work_values: np.ndarray,
    beta: float,
    clip: Optional[float] = None,
    max_clip: Optional[float] = None,
) -> float:
    """Stable Jarzynski free-energy estimator.

    Computes:

        ΔF ≈ -(1/β) log E[exp(-β W)]

    using a numerically stable log-sum-exp accumulation to avoid overflow
    when work values are large.

    Args:
        work_values: Per-trajectory cumulative work values W, shape (N,).
        beta: Inverse temperature.
        clip: Legacy alias for *max_clip*.  If both are provided, *max_clip*
            takes precedence.
        max_clip: If given, work values are clipped to the interval
            ``[-max_clip, max_clip]`` before exponentiation.  This prevents
            overflow on CIFAR-scale trajectories where individual work values
            can be very large.  Defaults to ``None`` (no clipping).

    Returns:
        Scalar estimate of the free-energy difference ΔF.
    """
    # max_clip takes precedence over the legacy clip parameter.
    effective_clip = max_clip if max_clip is not None else clip
    if effective_clip is not None:
        work_values = np.clip(work_values, -effective_clip, effective_clip)

    # log_weights = -β W
    # ΔF = -(1/β) * LogMeanExp(log_weights)
    #    = -(1/β) * [logsumexp(log_weights) - log(N)]
    log_weights = -beta * work_values
    n = len(log_weights)
    m = np.max(log_weights)
    # Numerically stable LogMeanExp:
    #   LogMeanExp(x) = logsumexp(x) - log(N) = m + log(sum(exp(x-m))) - log(N)
    # (audit fix CAT3-1: removed dead `log_z` variable from prior refactor)
    lse = float(np.log(np.sum(np.exp(log_weights - m))) + m)  # logsumexp
    log_mean_exp = lse - math.log(n)
    return -log_mean_exp / beta


def crooks_ratio(work_forward: np.ndarray, work_reverse: np.ndarray, beta: float) -> float:
    """Crooks fluctuation theorem ratio from forward/reverse work distributions."""
    eps = 1e-8
    hist_f, bins = np.histogram(work_forward, bins=50, density=True)
    hist_r, _ = np.histogram(-work_reverse, bins=bins, density=True)
    ratio = (hist_f + eps) / (hist_r + eps)
    midpoints = 0.5 * (bins[1:] + bins[:-1])
    return float(np.mean(np.log(ratio) + beta * midpoints))


def crooks_crossing_point(
    work_forward: list[float],
    work_reverse: list[float],
    bins: int = 80,
) -> float:
    """Estimate the Crooks crossing point W* where P_fwd(W*) = P_rev(-W*).

    At the crossing point W* = ΔF (the free energy difference). Finds W* by
    building normalised histograms of the forward and reverse (sign-flipped)
    work distributions and locating their intersection.
    """
    fwd = np.asarray(work_forward, dtype=np.float64)
    rev = np.asarray(work_reverse, dtype=np.float64)
    all_vals = np.concatenate([fwd, -rev])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if lo == hi:
        return float(lo)
    bin_edges = np.linspace(lo, hi, bins + 1)
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_f, _ = np.histogram(fwd, bins=bin_edges, density=True)
    hist_r, _ = np.histogram(-rev, bins=bin_edges, density=True)
    diff = hist_f - hist_r
    # Find sign change in diff → crossing point
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        # No crossing found — return midpoint between the two distribution means
        return float(0.5 * (fwd.mean() + (-rev).mean()))
    idx = sign_changes[len(sign_changes) // 2]  # take central crossing
    # Linear interpolation within the bin
    d0, d1 = diff[idx], diff[idx + 1]
    x0, x1 = midpoints[idx], midpoints[idx + 1]
    if abs(d1 - d0) < 1e-15:
        return float(0.5 * (x0 + x1))
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


class JarzynskiEstimator:
    """Stateful wrapper around :func:`jarzynski_work_estimate`."""

    def __init__(self, beta: float = 1.0, max_clip: Optional[float] = None) -> None:
        self.beta = beta
        self.max_clip = max_clip
        self._work_values: list[float] = []

    def record(self, work: float) -> None:
        self._work_values.append(float(work))

    def estimate(self) -> float:
        return jarzynski_work_estimate(
            np.array(self._work_values, dtype=np.float64),
            beta=self.beta,
            max_clip=self.max_clip,
        )

    def reset(self) -> None:
        self._work_values = []
