from __future__ import annotations

import numpy as np


def entropy_production_estimate(states: np.ndarray, drifts: np.ndarray, controls: np.ndarray, temperature: float, dt: float) -> float:
    """Discrete approximation of entropy production using Seifert formalism."""
    if drifts is None or controls is None:
        return 0.0
    integrand = np.sum((drifts + controls) * controls, axis=-1)
    return float(np.sum(integrand) * dt / max(temperature, 1e-6))
