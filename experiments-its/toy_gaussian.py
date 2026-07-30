from __future__ import annotations

from typing import Dict

import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

from ..controllers import ControlConfig, build_control_policy
from ..objectives import wasserstein_1d
from ..samplers import ControlledSDEConfig, ControlledSampler, PotentialConfig
from ..training import TrainingConfig


def _make_target_distribution(device: torch.device) -> MixtureSameFamily:
    mix = Categorical(torch.tensor([0.5, 0.5], device=device))
    components = Normal(torch.tensor([-2.0, 2.0], device=device), torch.tensor([0.7, 0.7], device=device))
    return MixtureSameFamily(mix, components)


def build_gaussian_mixture_experiment(device: str = "cpu") -> Dict[str, object]:
    device_obj = torch.device(device)
    potential_cfg = PotentialConfig(
        kind="gaussian_mixture",
        dim=1,
        mixture_means=[-2.0, 2.0],
        mixture_scales=[0.7, 0.7],
    )
    sampler = ControlledSampler(
        ControlledSDEConfig(dt=0.01, steps=450, diffusion=1.0, control_scale=0.5),
        potential_cfg,
    )
    control_cfg = ControlConfig(
        state_dim=potential_cfg.dim,
        hidden_dim=72,
        depth=3,
        time_embedding_dim=6,
        dropout=0.15,
    )
    controller = build_control_policy(control_cfg, device=device_obj)
    training_cfg = TrainingConfig(
        device=device,
        steps=200,
        lr=1.2e-3,
        control_weight=0.1,
        control_weight_final=0.24,
        drift_alignment_weight=0.18,
        drift_alignment_weight_final=0.32,
        entropy_weight=0.01,
        entropy_weight_final=0.03,
        occupancy_balance_weight=0.3,
        occupancy_balance_weight_final=0.6,
        occupancy_balance_target=0.0,
        matching_target_mean=0.0,
        matching_target_var=1.0,
        sampler_steps_initial=240,
        sampler_steps_final=450,
        temperature_initial=1.05,
        temperature_final=0.9,
        y0_noise_std=0.4,
        batch_size=2,
        enforce_symmetry=True,
        log_diagnostics_every=20,
        log_dir="logs",
        run_name="toy_gaussian",
    )
    target_dist = _make_target_distribution(device_obj)

    def quality_fn(samples: torch.Tensor, cfg: TrainingConfig) -> torch.Tensor:
        tail = samples[int(0.5 * samples.shape[0]) :]
        target = target_dist.sample((tail.shape[0],)).unsqueeze(-1)
        return wasserstein_1d(tail, target)

    return {
        "sampler": sampler,
        "controller": controller,
        "training": training_cfg,
        "quality_fn": quality_fn,
    }
