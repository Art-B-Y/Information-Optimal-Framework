from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from ..metrics import (
    occupancy_statistics,
    summarise_path,
    trajectory_diagnostics,
)
from ..objectives import control_energy, matching_moments
from ..utils import configure_logging, seed_everything

logger = logging.getLogger(__name__)


def _interp(start: float, end: Optional[float], alpha: float) -> float:
    if end is None:
        return start
    return (1.0 - alpha) * start + alpha * end


@dataclass
class TrainingConfig:
    steps: int = 200
    lr: float = 1e-3
    log_every: int = 10
    control_weight: float = 0.2
    drift_alignment_weight: float = 0.2
    entropy_weight: float = 0.0
    occupancy_balance_weight: float = 0.0
    control_weight_final: Optional[float] = None
    drift_alignment_weight_final: Optional[float] = None
    entropy_weight_final: Optional[float] = None
    occupancy_balance_weight_final: Optional[float] = None
    occupancy_balance_target: float = 0.0
    matching_target_mean: float = 0.0
    matching_target_var: float = 1.0
    device: str = "cpu"
    seed: int = 0
    y0: Optional[list[float]] = None
    y0_noise_std: float = 0.0
    sampler_steps_initial: Optional[int] = None
    sampler_steps_final: Optional[int] = None
    temperature_initial: Optional[float] = None
    temperature_final: Optional[float] = None
    batch_size: int = 1
    enforce_symmetry: bool = False
    log_diagnostics_every: int = 25
    optimizer_factory: Callable[[torch.nn.Module, float], torch.optim.Optimizer] = field(
        default=lambda model, lr: torch.optim.Adam(model.parameters(), lr=lr)
    )
    log_dir: Optional[str] = None
    run_name: Optional[str] = None
    log_file: Optional[str] = None


@dataclass
class _RolloutStats:
    control_energy: float = 0.0
    entropy_estimate: float = 0.0
    variance: float = 0.0
    occupancy_left: float = 0.0
    occupancy_right: float = 0.0
    occupancy_center: float = 0.0
    occupancy_balance: float = 0.0
    trajectory_spread: float = 0.0
    final_state_abs: float = 0.0
    balance_penalty: float = 0.0


def _resolve_log_file(config: TrainingConfig) -> Optional[str]:
    if config.log_file:
        return config.log_file
    if not config.log_dir:
        return None
    name = config.run_name or "toy_train"
    stamp = np.random.randint(0, 1_000_000)
    return str(Path(config.log_dir) / f"{name}_{stamp}.log")


def run_training(
    sampler,
    controller: torch.nn.Module,
    config: TrainingConfig,
    quality_fn: Optional[Callable[[torch.Tensor, TrainingConfig], torch.Tensor]] = None,
) -> Dict[str, float]:
    log_file = _resolve_log_file(config)
    configure_logging(log_file=log_file)
    device = torch.device(config.device)
    controller = controller.to(device)
    controller.train()
    seed_state = seed_everything(config.seed)
    optimizer = config.optimizer_factory(controller, config.lr)

    if config.y0 is None:
        y0 = np.zeros((sampler.dim,), dtype=np.float32)
    else:
        y0 = np.asarray(config.y0, dtype=np.float32)

    base_steps = config.sampler_steps_initial or sampler.config.steps
    final_steps = config.sampler_steps_final or base_steps
    base_temperature = config.temperature_initial or sampler.config.temperature
    final_temperature = config.temperature_final or base_temperature
    batch_size = max(1, int(config.batch_size))

    def make_y0(step_idx: int, batch_idx: int) -> np.ndarray:
        if config.y0_noise_std <= 0:
            return y0
        noise_seed = seed_state.seed + 7919 * step_idx + 104729 * batch_idx
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(loc=0.0, scale=config.y0_noise_std, size=y0.shape).astype(np.float32)
        return y0 + noise

    def quality_loss_fn(samples: torch.Tensor, cfg: TrainingConfig) -> torch.Tensor:
        if quality_fn is not None:
            return quality_fn(samples, cfg)
        return matching_moments(
            samples,
            target_mean=cfg.matching_target_mean,
            target_var=cfg.matching_target_var,
        )

    last_metrics: Dict[str, float] = {}
    for step in range(1, config.steps + 1):
        progress = 0.0 if config.steps <= 1 else (step - 1) / (config.steps - 1)
        c_weight = _interp(config.control_weight, config.control_weight_final, progress)
        d_weight = _interp(config.drift_alignment_weight, config.drift_alignment_weight_final, progress)
        e_weight = _interp(config.entropy_weight, config.entropy_weight_final, progress)
        b_weight = _interp(config.occupancy_balance_weight, config.occupancy_balance_weight_final, progress)
        eff_steps = max(1, int(round(_interp(base_steps, final_steps, progress))))
        eff_temp = _interp(base_temperature, final_temperature, progress)

        optimizer.zero_grad()
        total_loss_tensor = torch.zeros((), device=device)
        quality_sum = torch.zeros((), device=device)
        control_sum = torch.zeros((), device=device)
        drift_sum = torch.zeros((), device=device)
        entropy_penalty_sum = torch.zeros((), device=device)
        balance_penalty_sum = torch.zeros((), device=device)

        stats = _RolloutStats()
        rollout_count = 0
        symmetry_factors = [1.0, -1.0] if config.enforce_symmetry else [1.0]

        for batch_idx in range(batch_size):
            base_y0 = make_y0(step, batch_idx)
            for sign in symmetry_factors:
                signed_y0 = base_y0 * sign
                rollout_seed = seed_state.seed + step * 1009 + batch_idx * 53 + int(sign < 0)
                rollout = sampler.simulate(
                    signed_y0,
                    controller=controller,
                    seed=rollout_seed,
                    device=device,
                    steps_override=eff_steps,
                    temperature_override=eff_temp,
                )

                control_torch: torch.Tensor = rollout["control_torch"].to(device)
                states = torch.as_tensor(rollout["states"], dtype=torch.float32, device=device)
                drifts = rollout.get("drifts")
                drift_tensor = (
                    torch.as_tensor(drifts, dtype=torch.float32, device=device)
                    if drifts is not None
                    else torch.zeros_like(control_torch)
                )

                quality_loss = quality_loss_fn(states, config)
                control_loss = control_energy(control_torch)
                drift_alignment = torch.mean((control_torch + drift_tensor) ** 2)
                entropy_penalty = torch.tensor(float(rollout["entropy_estimate"]), device=device)

                occ_stats = occupancy_statistics(rollout["states"])
                balance_error = occ_stats["occupancy_balance"] - config.occupancy_balance_target
                balance_penalty_value = balance_error**2
                balance_penalty = torch.tensor(balance_penalty_value, device=device, dtype=torch.float32)

                total_loss_tensor = total_loss_tensor + (
                    quality_loss + c_weight * control_loss + d_weight * drift_alignment
                    + e_weight * entropy_penalty + b_weight * balance_penalty
                )
                quality_sum = quality_sum + quality_loss
                control_sum = control_sum + control_loss
                drift_sum = drift_sum + drift_alignment
                entropy_penalty_sum = entropy_penalty_sum + entropy_penalty
                balance_penalty_sum = balance_penalty_sum + balance_penalty

                summary = summarise_path(rollout)
                traj_stats = trajectory_diagnostics(rollout["states"])
                stats.control_energy += summary.control_energy
                stats.entropy_estimate += summary.entropy_estimate
                stats.variance += summary.variance
                stats.occupancy_left += occ_stats["occupancy_left"]
                stats.occupancy_right += occ_stats["occupancy_right"]
                stats.occupancy_center += occ_stats["occupancy_center"]
                stats.occupancy_balance += occ_stats["occupancy_balance"]
                stats.balance_penalty += balance_penalty_value
                stats.trajectory_spread += traj_stats["trajectory_spread"]
                stats.final_state_abs += traj_stats["final_state_abs"]

                rollout_count += 1

        assert rollout_count > 0
        total_loss = total_loss_tensor / rollout_count
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=10.0)
        optimizer.step()

        last_metrics = {
            "loss": float(total_loss.item()),
            "quality_loss": float((quality_sum / rollout_count).item()),
            "control_energy": stats.control_energy / rollout_count,
            "entropy_estimate": stats.entropy_estimate / rollout_count,
            "variance": stats.variance / rollout_count,
            "drift_alignment": float((drift_sum / rollout_count).item()),
            "occupancy_left": stats.occupancy_left / rollout_count,
            "occupancy_right": stats.occupancy_right / rollout_count,
            "occupancy_center": stats.occupancy_center / rollout_count,
            "occupancy_balance": stats.occupancy_balance / rollout_count,
            "balance_penalty": stats.balance_penalty / rollout_count,
            "trajectory_spread": stats.trajectory_spread / rollout_count,
            "final_state_abs": stats.final_state_abs / rollout_count,
            "control_weight": float(c_weight),
            "drift_weight": float(d_weight),
            "entropy_weight": float(e_weight),
            "balance_weight": float(b_weight),
            "effective_steps": float(eff_steps),
            "effective_temperature": float(eff_temp),
            "batch_size": float(rollout_count),
        }

        if step % config.log_every == 0 or step == 1:
            logger.info(
                (
                    "step=%d loss=%.4f quality=%.4f control=%.4f entropy=%.4f "
                    "left=%.2f right=%.2f spread=%.3f bal=%.2f"
                ),
                step,
                last_metrics["loss"],
                last_metrics["quality_loss"],
                last_metrics["control_energy"],
                last_metrics["entropy_estimate"],
                last_metrics["occupancy_left"],
                last_metrics["occupancy_right"],
                last_metrics["trajectory_spread"],
                last_metrics["occupancy_balance"],
            )

        if config.log_diagnostics_every and step % config.log_diagnostics_every == 0:
            logger.info(
                (
                    "diagnostics step=%d balance=%.2f pen=%.4f final_abs=%.3f "
                    "c_w=%.3f d_w=%.3f b_w=%.3f temp=%.3f steps=%d batch=%.0f"
                ),
                step,
                last_metrics["occupancy_balance"],
                last_metrics["balance_penalty"],
                last_metrics["final_state_abs"],
                last_metrics["control_weight"],
                last_metrics["drift_weight"],
                last_metrics["balance_weight"],
                last_metrics["effective_temperature"],
                int(last_metrics["effective_steps"]),
                last_metrics["batch_size"],
            )

    return last_metrics
