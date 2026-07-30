from __future__ import annotations

import numpy as np


def occupancy_statistics(states: np.ndarray, threshold: float = 0.2) -> dict[str, float]:
    if states.ndim == 1:
        tail = states
    else:
        window = min(200, states.shape[0])
        tail = states[-window:, 0]
    left = float(np.mean(tail < -threshold))
    right = float(np.mean(tail > threshold))
    center = float(np.mean(np.abs(tail) <= threshold))
    balance = right - left
    return {
        "occupancy_left": left,
        "occupancy_right": right,
        "occupancy_center": center,
        "occupancy_balance": balance,
    }


def trajectory_diagnostics(states: np.ndarray) -> dict[str, float]:
    spread = float(np.std(states, axis=0).mean())
    final_state = states[-1]
    final_abs = float(np.abs(final_state).mean())
    return {
        "trajectory_spread": spread,
        "final_state_abs": final_abs,
    }
