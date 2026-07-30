from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PathSummary:
    control_energy: float
    entropy_estimate: float
    variance: float
    final_state_mean: float
    drift_correlation: Optional[float]


def summarise_path(sim_result: dict[str, np.ndarray]) -> PathSummary:
    states = sim_result["states"]
    controls = sim_result["controls"]
    drifts = sim_result.get("drifts")
    control_energy = float(sim_result["control_energy"])
    entropy_estimate = float(sim_result["entropy_estimate"])
    variance = float(np.var(states, axis=0).mean())
    final_state_mean = float(states[-1].mean())
    drift_correlation = None
    if drifts is not None:
        numerator = float(np.sum(controls * drifts))
        denom = float(np.linalg.norm(controls) * np.linalg.norm(drifts) + 1e-8)
        drift_correlation = numerator / denom
    return PathSummary(
        control_energy=control_energy,
        entropy_estimate=entropy_estimate,
        variance=variance,
        final_state_mean=final_state_mean,
        drift_correlation=drift_correlation,
    )
