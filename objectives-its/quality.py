from __future__ import annotations

import torch


def matching_moments(samples: torch.Tensor, target_mean: float = 0.0, target_var: float = 1.0) -> torch.Tensor:
    mean = torch.mean(samples, dim=0)
    var = torch.var(samples, dim=0, unbiased=False)
    loss_mean = torch.sum((mean - target_mean) ** 2)
    loss_var = torch.sum((var - target_var) ** 2)
    return loss_mean + loss_var


def wasserstein_1d(samples: torch.Tensor, target_samples: torch.Tensor) -> torch.Tensor:
    sorted_src, _ = torch.sort(samples.squeeze())
    sorted_tgt, _ = torch.sort(target_samples.squeeze())
    min_len = min(sorted_src.shape[0], sorted_tgt.shape[0])
    return torch.mean(torch.abs(sorted_src[:min_len] - sorted_tgt[:min_len]))
