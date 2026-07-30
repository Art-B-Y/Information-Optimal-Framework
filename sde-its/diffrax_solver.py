from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
    import diffrax
except ModuleNotFoundError:  # pragma: no cover
    jax = None
    diffrax = None


@dataclass
class DiffraxConfig:
    dt: float = 1e-3
    steps: int = 100
    solver: str = "euler"  # or "heun"


def _make_solver(name: str):
    if diffrax is None:
        raise ImportError("diffrax is not installed. Please `pip install diffrax` to use the JAX solver.")
    if name == "heun":
        return diffrax.Heun()
    return diffrax.Euler()


def sample_with_diffrax(
    drift_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    diffusion_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    y0: jnp.ndarray,
    cfg: DiffraxConfig,
    key: Optional["jax.random.PRNGKey"] = None,
) -> jnp.ndarray:
    if diffrax is None or jax is None:
        raise ImportError("diffrax or jax not installed. Please install to use the JAX solver.")
    solver = _make_solver(cfg.solver)
    term = diffrax.MultiTerm(
        diffrax.ODETerm(drift_fn),
        diffrax.ControlTerm(diffusion_fn, diffrax.UnsafeBrownianPath(shape=y0.shape, key=key)),
    )
    saveat = diffrax.SaveAt(t1=True)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=cfg.dt * cfg.steps,
        dt0=cfg.dt,
        y0=y0,
        saveat=saveat,
    )
    return sol.ys
