from __future__ import annotations

from typing import Dict

import torch

from ..controllers import ControlConfig, build_control_policy
from ..samplers import ControlledSDEConfig, ControlledSampler, PotentialConfig
from ..training import TrainingConfig


def build_double_well_experiment(device: str = "cpu") -> Dict[str, object]:
    device_obj = torch.device(device)
    potential_cfg = PotentialConfig(kind="double_well", dim=1, well_depth=3.0)
    sampler = ControlledSampler(
        ControlledSDEConfig(dt=0.005, steps=600, diffusion=0.8, control_scale=0.6),
        potential_cfg,
    )
    control_cfg = ControlConfig(
        state_dim=potential_cfg.dim,
        hidden_dim=96,
        depth=3,
        time_embedding_dim=8,
        dropout=0.1,
    )
    controller = build_control_policy(control_cfg, device=device_obj)
    training_cfg = TrainingConfig(
        device=device,
        steps=240,
        lr=7e-4,
        control_weight=0.12,
        control_weight_final=0.28,
        drift_alignment_weight=0.2,
        drift_alignment_weight_final=0.4,
        entropy_weight=0.02,
        entropy_weight_final=0.04,
        occupancy_balance_weight=0.35,
        occupancy_balance_weight_final=0.8,
        occupancy_balance_target=0.0,
        matching_target_mean=0.0,
        matching_target_var=1.3,
        sampler_steps_initial=280,
        sampler_steps_final=600,
        temperature_initial=1.1,
        temperature_final=0.85,
        y0_noise_std=0.5,
        batch_size=2,
        enforce_symmetry=True,
        log_diagnostics_every=20,
        log_dir="logs",
        run_name="double_well",
    )

    def quality_fn(samples: torch.Tensor, _: TrainingConfig) -> torch.Tensor:
        tail = samples[int(0.75 * samples.shape[0]) :]
        return torch.mean((tail.abs() - 1.0) ** 2)

    return {
        "sampler": sampler,
        "controller": controller,
        "training": training_cfg,
        "quality_fn": quality_fn,
    }
