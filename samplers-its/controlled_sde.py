from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ..utils import array_to_torch, ensure_numpy


@dataclass
class PotentialConfig:
    kind: str = "harmonic"
    dim: int = 1
    well_depth: float = 4.0
    mixture_means: Optional[list[float]] = None
    mixture_scales: Optional[list[float]] = None


@dataclass
class ControlledSDEConfig:
    dt: float = 0.01
    steps: int = 1_000
    diffusion: float = 1.0
    control_scale: float = 1.0
    temperature: float = 1.0
    record_drift: bool = True
    solver: str = "euler"  # options: euler, heun, milstein


def harmonic_potential(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum(x**2)


def double_well_potential(x: jnp.ndarray, depth: float = 4.0) -> jnp.ndarray:
    return depth * (0.25 * jnp.sum(x**4) - 0.5 * jnp.sum(x**2))


def gaussian_mixture_potential(
    x: jnp.ndarray,
    means: jnp.ndarray,
    scales: jnp.ndarray,
) -> jnp.ndarray:
    diff = (x - means) / scales
    exponent = -0.5 * jnp.sum(diff**2, axis=-1)
    log_z = jnp.log(jnp.sum(jnp.exp(exponent)))
    return -log_z


def make_potential(config: PotentialConfig) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if config.kind == "harmonic":
        return harmonic_potential
    if config.kind == "double_well":
        return lambda x: double_well_potential(x, depth=config.well_depth)
    if config.kind == "gaussian_mixture":
        means = jnp.asarray(config.mixture_means or [-2.0, 2.0], dtype=jnp.float32)
        if means.ndim == 1:
            means = means[:, None]
        scales = jnp.asarray(config.mixture_scales or [1.0] * means.shape[0], dtype=jnp.float32)
        if scales.ndim == 1:
            scales = scales[:, None]
        return lambda x: gaussian_mixture_potential(x[None, :], means, scales)
    raise ValueError(f"Unsupported potential kind: {config.kind}")


class ControlledSampler:
    def __init__(self, sde_config: ControlledSDEConfig, potential_config: PotentialConfig) -> None:
        self.config = sde_config
        self.potential_cfg = potential_config
        self.potential_fn = make_potential(potential_config)
        self.grad_potential = jax.grad(lambda y: self.potential_fn(jnp.asarray(y, dtype=jnp.float32)))

    @property
    def dim(self) -> int:
        return self.potential_cfg.dim

    def _euler_torch(
        self,
        y0: torch.Tensor,
        controller: Optional[torch.nn.Module] = None,
        seed: int = 0,
        device: Optional[torch.device] = None,
        steps_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        dt_override: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """Fully differentiable Euler-Maruyama sampler using pure PyTorch tensors.

        Unlike ``simulate()``, this method keeps the state ``x`` as a
        ``torch.Tensor`` throughout, so gradients flow through the entire
        trajectory.  Use this path during training when you need to
        backpropagate a quality objective through the stochastic trajectory.

        The JAX-based potential gradient is evaluated by converting to/from
        NumPy at each step; this is the only non-differentiable boundary.  A
        fully differentiable potential would replace that with a PyTorch
        autograd call.

        Returns a dict with keys:
            ``x``            – final state tensor, shape ``(dim,)``
            ``states``       – trajectory tensor, shape ``(steps+1, dim)``
            ``controls``     – control tensors, shape ``(steps, dim)``
            ``control_energy`` – scalar tensor
            ``entropy_estimate`` – scalar tensor
        """
        cfg = self.config
        dt = dt_override if dt_override is not None else cfg.dt
        steps = int(steps_override) if steps_override is not None else cfg.steps
        total_time = dt * steps
        temperature = temperature_override if temperature_override is not None else cfg.temperature

        if device is None:
            device = y0.device if isinstance(y0, torch.Tensor) else torch.device("cpu")

        x: torch.Tensor = y0.to(device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, dim) for batch-dim consistency

        times = torch.linspace(0.0, total_time, steps + 1, device=device)
        states_list: list[torch.Tensor] = [x.detach().clone()]
        controls_list: list[torch.Tensor] = []
        control_energy = torch.zeros(1, device=device)
        entropy_estimate = torch.zeros(1, device=device)

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        for idx in range(steps):
            t = times[idx].item()
            # Compute gradient of potential via JAX (not differentiable through
            # the JAX call itself, but the control output remains differentiable).
            # TODO: replace with a PyTorch-native potential for full differentiability.
            x_np = x.detach().cpu().numpy().squeeze(0)
            grad_np = np.asarray(self.grad_potential(jnp.asarray(x_np, dtype=jnp.float32)), dtype=np.float32)
            drift = -torch.from_numpy(grad_np).to(device).unsqueeze(0)  # (1, dim)

            noise_std = float(np.sqrt(2.0 * temperature * cfg.diffusion * dt))
            noise = torch.randn_like(x, generator=rng) * noise_std

            control_vec = torch.zeros_like(x)
            if controller is not None:
                tau = torch.tensor([[t / total_time if total_time > 0 else 0.0]], device=device)
                control_out = controller(x, tau)
                control_vec = control_out

            x = x + (drift + cfg.control_scale * control_vec) * dt + noise
            states_list.append(x.detach().clone())
            controls_list.append(control_vec)
            control_energy = control_energy + (control_vec.pow(2).sum() * dt)
            entropy_estimate = entropy_estimate + (drift * control_vec).sum() * dt

        return {
            "x": x.squeeze(0),
            "states": torch.cat([s for s in states_list], dim=0),  # (steps+1, dim)
            "controls": torch.stack(controls_list, dim=0).squeeze(1),  # (steps, dim)
            "control_energy": control_energy.squeeze(0),
            "entropy_estimate": entropy_estimate.squeeze(0),
        }

    def simulate(
        self,
        y0: np.ndarray,
        controller: Optional[torch.nn.Module] = None,
        seed: int = 0,
        device: Optional[torch.device] = None,
        steps_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        dt_override: Optional[float] = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """NumPy/JAX-based Euler-Maruyama simulator.

        NOTE: The state ``x`` is converted to NumPy at every step via
        ``.detach().cpu().numpy()``.  This **breaks the PyTorch computational
        graph**, so the quality objective cannot backpropagate through the full
        stochastic trajectory.  The toy training loop can only optimise scalar
        control penalties computed on the Torch side (e.g. control energy).

        TODO: Replace this method with a fully Torch-native loop for end-to-end
        differentiability.  Use ``_euler_torch()`` in training code that needs
        ``requires_grad=True`` paths through the trajectory.
        """
        cfg = self.config
        dt = dt_override if dt_override is not None else cfg.dt
        steps = int(steps_override) if steps_override is not None else cfg.steps
        total_time = dt * steps
        temperature = temperature_override if temperature_override is not None else cfg.temperature
        dim = y0.shape[-1]
        times = np.linspace(0.0, total_time, steps + 1, dtype=np.float32)
        state = ensure_numpy(y0).astype(np.float32)
        states = np.zeros((steps + 1, dim), dtype=np.float32)
        controls = np.zeros((steps, dim), dtype=np.float32)
        drifts = np.zeros((steps, dim), dtype=np.float32)
        control_history: list[torch.Tensor] = []
        control_energy = 0.0
        entropy_estimate = 0.0
        rng = np.random.default_rng(seed)

        states[0] = state
        for idx in range(steps):
            t = times[idx]
            drift = -np.asarray(self.grad_potential(jnp.asarray(state)), dtype=np.float32)
            noise_std = np.sqrt(2.0 * temperature * cfg.diffusion * dt)
            noise = noise_std * rng.normal(size=dim).astype(np.float32)
            control_np = np.zeros_like(state)
            control_torch = torch.zeros(dim, device=device, dtype=torch.float32)
            if controller is not None:
                xt = array_to_torch(state[None, :], device=device)
                tau = np.array([[t / total_time if total_time > 0 else 0.0]], dtype=np.float32)
                t_tensor = array_to_torch(tau, device=device)
                control_out = controller(xt, t_tensor)
                control_torch = control_out.squeeze(0)
                control_np = control_torch.detach().cpu().numpy()
            if cfg.solver == "heun":
                # Predictor
                state_pred = state + (drift + cfg.control_scale * control_np) * dt + noise
                drift_pred = -np.asarray(self.grad_potential(jnp.asarray(state_pred)), dtype=np.float32)
                drift_avg = 0.5 * (drift + drift_pred)
                state = state + (drift_avg + cfg.control_scale * control_np) * dt + noise
            elif cfg.solver == "milstein":
                # Simplified Milstein (diagonal noise)
                state = state + (drift + cfg.control_scale * control_np) * dt + noise + 0.5 * (noise_std**2) * (
                    np.sign(noise) * np.sqrt(dt)
                )
            else:
                state = state + (drift + cfg.control_scale * control_np) * dt + noise
            states[idx + 1] = state
            controls[idx] = control_np
            drifts[idx] = drift
            control_history.append(control_torch)
            control_energy += float(np.dot(control_np, control_np)) * dt
            entropy_estimate += float(np.dot(drift, control_np)) * dt

        control_stack = torch.stack(control_history) if control_history else torch.zeros((steps, dim))

        return {
            "times": times,
            "states": states,
            "controls": controls,
            "control_torch": control_stack,
            "drifts": drifts if cfg.record_drift else None,
            "control_energy": np.array(control_energy, dtype=np.float32),
            "entropy_estimate": np.array(entropy_estimate, dtype=np.float32),
        }


# Alias for API compatibility.
ControlledSDESampler = ControlledSampler
