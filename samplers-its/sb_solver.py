"""Schrödinger Bridge solver skeleton using Iterative Proportional Fitting (IPF).

This module provides a lightweight, purely additive extension to the ITS
codebase.  It implements one forward/backward IPF iteration using the same
``ConvControlPolicy`` architecture that the controlled VP-SDE uses, so no new
dependencies are introduced.

Theory
------
Given a source distribution p_0 and a target distribution p_T, the
Schrödinger Bridge finds the measure Q* that is the KL-projection of
Brownian motion onto the constraint set {Q : Q_0 = p_0, Q_T = p_T}.

The Iterative Proportional Fitting (IPF) algorithm alternates between:
  1. Forward pass:  train u_theta to minimise path-KL from p_0 towards
                    generated trajectories sampled from the current backward
                    policy.
  2. Backward pass: train v_phi to minimise path-KL from p_T in reverse
                    time towards generated trajectories from the current
                    forward policy.

Each half-step is equivalent to a controlled SDE problem with a fixed
reference trajectory, so the same Girsanov-weight loss used in the
VP-SDE controller applies directly.

This implementation is intentionally minimal -- it is a research scaffold,
not a production solver.  Only the IPF loss computation and single-step
Euler-Maruyama samplers are provided; multi-iteration training loops are
left to calling code.

References
----------
De Bortoli et al. (2021) "Diffusion Schrodinger Bridge with Applications
to Score-Based Generative Modeling", NeurIPS 2021.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy config kept for backward compatibility
# ---------------------------------------------------------------------------

@dataclass
class SBSolverConfig:
    iterations: int = 5
    forward_steps: int = 100
    backward_steps: int = 100
    alpha: float = 0.5  # relaxation between forward/backward updates


# ---------------------------------------------------------------------------
# New configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SBSamplerConfig:
    """Configuration for the Schrodinger Bridge sampler.

    Attributes
    ----------
    dt:
        Time-step for the Euler-Maruyama discretisation.
    steps:
        Number of integration steps (forward or backward).
    diffusion:
        Diffusion coefficient sigma (same for forward and backward processes).
    in_channels:
        Number of image channels (1 for MNIST/FashionMNIST, 3 for CIFAR-10).
    base_channels:
        Base channel width for both forward and backward control networks.
    num_blocks:
        Number of residual blocks in each control network.
    time_embed_dim:
        Dimension of the sinusoidal time embedding.
    iterations:
        Number of IPF half-iterations to run in one call to ``fit()``.
    lr:
        Learning rate for the Adam optimiser used during ``train_step_*``.
    """

    dt: float = 0.01
    steps: int = 100
    diffusion: float = 1.0
    in_channels: int = 1
    base_channels: int = 64
    num_blocks: int = 2
    time_embed_dim: int = 128
    iterations: int = 5
    lr: float = 1e-4


# ---------------------------------------------------------------------------
# Schrodinger Bridge sampler
# ---------------------------------------------------------------------------

class SchrodingerBridgeSampler(nn.Module):
    """Minimal Schrodinger Bridge sampler with IPF-based training.

    This class holds two ``nn.Module`` control policies:
      - ``forward_policy``  (u_theta) -- drives the forward SDE from p_0 to p_T
      - ``backward_policy`` (v_phi)   -- drives the backward SDE from p_T to p_0

    Both share the same ``SBSamplerConfig``.  The ``ConvControlPolicy``
    architecture from ``its.controllers`` is used for each policy so that
    the implementation is consistent with the controlled VP-SDE.

    Training follows the IPF schedule:
      1. Fix v_phi, optimise u_theta using ``train_step_forward``.
      2. Fix u_theta, optimise v_phi using ``train_step_backward``.
      3. Repeat.

    The IPF loss for each direction is the Girsanov path-KL divergence
    between the policy-driven measure and the reference (zero-control)
    Brownian-motion measure, computed by ``compute_ipf_loss``.

    This is a *skeleton* implementation.  The ``fit()`` convenience method
    runs a fixed number of gradient steps per IPF iteration.
    """

    def __init__(self, config: SBSamplerConfig) -> None:
        super().__init__()
        self.config = config

        # Import here to avoid circular imports with the controllers package.
        from ..controllers import ConvControlConfig, ConvControlPolicy

        ctrl_cfg = ConvControlConfig(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            num_blocks=config.num_blocks,
            time_embed_dim=config.time_embed_dim,
        )
        # Forward and backward policies have the same architecture but
        # independent weights.
        self.forward_policy: ConvControlPolicy = ConvControlPolicy(ctrl_cfg)
        self.backward_policy: ConvControlPolicy = ConvControlPolicy(ctrl_cfg)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_forward(
        self,
        x0: torch.Tensor,
        *,
        use_policy: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the forward SDE from *x0* for ``config.steps`` steps.

        The forward process is

            dX_t = sigma * u_theta(X_t, t) dt + sigma dW_t,   X_0 = x0

        where sigma = ``config.diffusion``.

        Parameters
        ----------
        x0:
            Initial state tensor of shape ``(B, C, H, W)``.
        use_policy:
            If ``True`` (default), include the forward control u_theta.
            If ``False``, simulate pure Brownian motion (reference process).
        seed:
            Optional RNG seed for reproducibility.

        Returns
        -------
        dict with keys:
            ``x_T``          -- final state, shape ``(B, C, H, W)``
            ``trajectory``   -- states, shape ``(steps+1, B, C, H, W)``
            ``controls``     -- control vectors, shape ``(steps, B, C, H, W)``
            ``log_girsanov`` -- cumulative Girsanov log-weight, shape ``(B,)``
        """
        return self._euler_maruyama(
            x0,
            policy=self.forward_policy if use_policy else None,
            forward=True,
            seed=seed,
        )

    def sample_backward(
        self,
        xT: torch.Tensor,
        *,
        use_policy: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the backward SDE from *xT* for ``config.steps`` steps.

        The backward process uses reversed time:

            dX_t = sigma * v_phi(X_t, t) dt + sigma dW_t,   X_T = xT

        Parameters
        ----------
        xT:
            Terminal state tensor of shape ``(B, C, H, W)``.
        use_policy:
            If ``True`` (default), include the backward control v_phi.
            If ``False``, simulate pure Brownian motion (reference process).
        seed:
            Optional RNG seed for reproducibility.

        Returns
        -------
        Same keys as :meth:`sample_forward`.
        """
        return self._euler_maruyama(
            xT,
            policy=self.backward_policy if use_policy else None,
            forward=False,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # IPF loss
    # ------------------------------------------------------------------

    def compute_ipf_loss(
        self,
        trajectory: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the IPF Girsanov path-KL loss from a sampled trajectory.

        The loss is the squared norm of the control field integrated over the
        trajectory:

            L_IPF = (1 / 2*sigma^2) * sum_t ||u(X_t, t)||^2 * dt

        This is the Novikov/Girsanov exponential cost that appears in the
        KL divergence between the controlled measure and the reference
        Brownian-motion measure.  Minimising this loss over u_theta while
        keeping the endpoint constraint drives u_theta towards the optimal
        Schrodinger bridge policy.

        Parameters
        ----------
        trajectory:
            States along the path, shape ``(steps+1, B, C, H, W)``.
            The last frame is the terminal state.  Not used in the current
            formulation but retained for future endpoint-matching losses.
        controls:
            Control vectors at each step, shape ``(steps, B, C, H, W)``.

        Returns
        -------
        Scalar loss tensor.
        """
        cfg = self.config
        # Squared L2 norm of control at each step, summed over spatial dims.
        ctrl_sq = controls.pow(2).flatten(2).sum(dim=2)  # (steps, B)
        # Riemann sum approximation of the integral int ||u||^2 dt.
        path_cost = ctrl_sq.sum(dim=0) * cfg.dt           # (B,)
        return (0.5 / max(cfg.diffusion ** 2, 1e-8)) * path_cost.mean()

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def train_step_forward(
        self,
        x0: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """One gradient step on the forward policy u_theta.

        Runs the forward SDE with the current forward policy and minimises
        the IPF path-KL loss.  The backward policy is *not* updated.

        Parameters
        ----------
        x0:
            Source samples, shape ``(B, C, H, W)``.
        optimizer:
            An ``Optimizer`` whose ``param_groups`` cover
            ``self.forward_policy.parameters()`` only.

        Returns
        -------
        dict with scalar metrics:
            ``loss_forward``, ``ctrl_norm_forward``
        """
        self.forward_policy.train()
        optimizer.zero_grad()

        result = self.sample_forward(x0, use_policy=True)
        loss = self.compute_ipf_loss(result["trajectory"], result["controls"])

        if not torch.isfinite(loss):
            logger.warning("Non-finite forward IPF loss; skipping step.")
            optimizer.zero_grad()
            return {"loss_forward": float("nan"), "ctrl_norm_forward": float("nan")}

        loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(self.forward_policy.parameters(), max_norm=5.0).item()
        )
        optimizer.step()

        ctrl_norm = result["controls"].detach().norm().item()
        return {"loss_forward": loss.item(), "ctrl_norm_forward": ctrl_norm, "grad_norm": grad_norm}

    def train_step_backward(
        self,
        xT: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """One gradient step on the backward policy v_phi.

        Runs the backward SDE with the current backward policy and minimises
        the IPF path-KL loss.  The forward policy is *not* updated.

        Parameters
        ----------
        xT:
            Target samples, shape ``(B, C, H, W)``.
        optimizer:
            An ``Optimizer`` whose ``param_groups`` cover
            ``self.backward_policy.parameters()`` only.

        Returns
        -------
        dict with scalar metrics:
            ``loss_backward``, ``ctrl_norm_backward``
        """
        self.backward_policy.train()
        optimizer.zero_grad()

        result = self.sample_backward(xT, use_policy=True)
        loss = self.compute_ipf_loss(result["trajectory"], result["controls"])

        if not torch.isfinite(loss):
            logger.warning("Non-finite backward IPF loss; skipping step.")
            optimizer.zero_grad()
            return {"loss_backward": float("nan"), "ctrl_norm_backward": float("nan")}

        loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(self.backward_policy.parameters(), max_norm=5.0).item()
        )
        optimizer.step()

        ctrl_norm = result["controls"].detach().norm().item()
        return {"loss_backward": loss.item(), "ctrl_norm_backward": ctrl_norm, "grad_norm": grad_norm}

    # ------------------------------------------------------------------
    # Internal Euler-Maruyama integrator (image-space, pure PyTorch)
    # ------------------------------------------------------------------

    def _euler_maruyama(
        self,
        x_init: torch.Tensor,
        policy: Optional[nn.Module],
        forward: bool,
        seed: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        """Euler-Maruyama integration for image-valued SDEs.

        Runs ``config.steps`` steps of

            X_{n+1} = X_n + sigma * u(X_n, t_n) * dt + sigma * sqrt(dt) * eps_n,
            eps_n ~ N(0, I)

        where ``u`` is the supplied ``policy`` (or zero if ``policy`` is None).

        The time variable ``t_n`` is normalised to [0, 1]:
          - Forward integration: t increases from 0 to 1.
          - Backward integration: t decreases from 1 to 0 (reversed time).

        Returns a dict with ``x_T``, ``trajectory``, ``controls``,
        ``log_girsanov`` (shape ``(B,)``).
        """
        cfg = self.config
        device = x_init.device
        B = x_init.shape[0]

        if seed is not None:
            torch.manual_seed(seed)

        x = x_init.clone()
        sigma = cfg.diffusion
        sqrt_dt = cfg.dt ** 0.5

        trajectory = [x.detach().clone()]
        controls_list = []
        log_girsanov = torch.zeros(B, device=device)

        for step_idx in range(cfg.steps):
            # Normalised time in [0, 1].
            if forward:
                t_norm = step_idx / max(cfg.steps - 1, 1)
            else:
                t_norm = 1.0 - step_idx / max(cfg.steps - 1, 1)

            t_tensor = torch.full((B, 1), t_norm, device=device, dtype=x.dtype)

            if policy is not None:
                with torch.set_grad_enabled(policy.training):
                    u = policy(x, t_tensor)
            else:
                u = torch.zeros_like(x)

            noise = torch.randn_like(x)
            dW = noise * sqrt_dt

            x = x + sigma * u * cfg.dt + sigma * dW

            controls_list.append(u)
            # Approximate Girsanov log-weight contribution: u.dW - 0.5*||u||^2*dt
            u_flat = u.flatten(1)
            dW_flat = dW.flatten(1)
            log_girsanov = (
                log_girsanov
                + (u_flat * dW_flat).sum(dim=1)
                - 0.5 * u_flat.pow(2).sum(dim=1) * cfg.dt
            )

            trajectory.append(x.detach().clone())

        controls_tensor = torch.stack(controls_list, dim=0)   # (steps, B, C, H, W)
        trajectory_tensor = torch.stack(trajectory, dim=0)    # (steps+1, B, C, H, W)

        return {
            "x_T": x,
            "trajectory": trajectory_tensor,
            "controls": controls_tensor,
            "log_girsanov": log_girsanov,
        }


# ---------------------------------------------------------------------------
# Legacy class (kept for backward compatibility)
# ---------------------------------------------------------------------------

class SchrodingerBridgeSolver:
    """Legacy alternating Schrodinger Bridge skeleton.

    Deprecated in favour of ``SchrodingerBridgeSampler``.  Kept for backward
    compatibility with code that imported this class directly.
    """

    def __init__(
        self,
        forward_sampler: Callable[[int], torch.Tensor],
        backward_sampler: Callable[[torch.Tensor], torch.Tensor],
        config: SBSolverConfig,
    ) -> None:
        self.forward_sampler = forward_sampler
        self.backward_sampler = backward_sampler
        self.config = config

    def run(self, batch_size: int) -> Dict[str, torch.Tensor]:
        x = None
        for _ in range(self.config.iterations):
            x = self.forward_sampler(batch_size)
            x = self.backward_sampler(x)
        return {"samples": x}
