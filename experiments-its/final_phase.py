from __future__ import annotations

from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf

from ..controllers import ControlConfig
from ..data import DatasetConfig
from ..models import ScoreUNetConfig, build_score_model
from ..sde import ScoreSDEConfig, ScoreSDESimulator
from ..training import ScoreTrainingConfig


def _resolve_device(device: str | None) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _to_dict(cfg: DictConfig | dict | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def build_final_phase_experiment(
    device: str | None = "cuda",
    dataset: DictConfig | dict | None = None,
    model: DictConfig | dict | None = None,
    score_training: DictConfig | dict | None = None,
    sampler: DictConfig | dict | None = None,
) -> Dict[str, object]:
    resolved_device = _resolve_device(device)

    dataset_base = {
        "name": "cifar10",
        "batch_size": 64,
        "num_workers": 0,
        "augment": True,
    }
    dataset_base.update(_to_dict(dataset))

    score_training_overrides = _to_dict(score_training)
    dataset_overrides = score_training_overrides.pop("dataset", {})
    model_overrides_from_training = score_training_overrides.pop("model", {})

    dataset_cfg = DatasetConfig(**{**dataset_base, **dataset_overrides})

    model_base = {
        "in_channels": 3,
        "base_channels": 64,
        "channel_mults": (1, 2, 2, 4),
    }
    model_base.update(_to_dict(model))
    model_base.update(model_overrides_from_training)
    model_cfg = ScoreUNetConfig(**model_base)

    training_kwargs = {
        "epochs": 10,
        "lr": 2e-4,
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "grad_clip": 1.0,
        "device": resolved_device,
        "log_interval": 50,
        "dataset": dataset_cfg,
        "model": model_cfg,
        "ema_decay": 0.999,
        "checkpoint_dir": "checkpoints/score",
        "save_interval": 1,
    }
    training_kwargs.update(score_training_overrides)
    training_cfg = ScoreTrainingConfig(**training_kwargs)

    sampler_kwargs = {
        "beta_min": 0.1,
        "beta_max": 20.0,
        "num_steps": 1000,
        "control_weight": 0.05,
    }
    sampler_kwargs.update(_to_dict(sampler))
    sde_cfg = ScoreSDEConfig(**sampler_kwargs)

    score_model = build_score_model(model_cfg).to(resolved_device)
    sampler_obj = ScoreSDESimulator(score_model, sde_cfg)
    control_cfg = ControlConfig(
        state_dim=score_model.config.in_channels,
        hidden_dim=256,
        depth=4,
        time_embedding_dim=16,
        dropout=0.2,
    )
    return {
        "dataset_cfg": dataset_cfg,
        "model": score_model,
        "sampler": sampler_obj,
        "score_training": training_cfg,
        "control_cfg": control_cfg,
    }
