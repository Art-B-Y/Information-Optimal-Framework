from __future__ import annotations

from typing import Dict

import torch

from ..controllers import ControlConfig, build_control_policy
from ..models import ScoreUNetConfig, build_score_model
from ..sde import ScoreSDEConfig
from ..training.controlled_score_training import ControlledScoreTrainingConfig


def build_controlled_mnist_experiment(
    device: str = "cpu",
    model: Dict | None = None,
    control: Dict | None = None,
    sde: Dict | None = None,
    training: Dict | None = None,
) -> Dict[str, object]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=32, channel_mults=(1, 2, 2))
    # 28x28 grayscale -> 784 dimensional flattened state
    control_cfg = ControlConfig(state_dim=28 * 28, hidden_dim=128, depth=3, time_embedding_dim=8)
    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=50, clamp=5.0, control_weight=1.0)

    if model:
        for k, v in model.items():
            setattr(model_cfg, k, v)
    if control:
        for k, v in control.items():
            setattr(control_cfg, k, v)
    if sde:
        for k, v in sde.items():
            setattr(sde_cfg, k, v)

    training_cfg = ControlledScoreTrainingConfig(
        device=device,
        epochs=5,
        batch_size=64,
        dataset_name="fashionmnist",
        model=model_cfg,
        control=control_cfg,
        sde=sde_cfg,
        control_weight=1e-2,
        log_interval=50,
    )
    if training:
        for k, v in training.items():
            setattr(training_cfg, k, v)

    score_model = build_score_model(model_cfg).to(device)
    control_policy = build_control_policy(control_cfg, device=device)
    return {
        "model": score_model,
        "control": control_policy,
        "training": training_cfg,
        "sde": sde_cfg,
    }
