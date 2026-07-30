from __future__ import annotations

from typing import Sequence

import torch


def denormalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo per-channel normalization."""
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    return images * std_t + mean_t


def clamp_unit(images: torch.Tensor) -> torch.Tensor:
    return images.clamp(0.0, 1.0)
