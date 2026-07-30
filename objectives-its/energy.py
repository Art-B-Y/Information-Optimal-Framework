from __future__ import annotations

import torch


def control_energy(control: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(control**2, dim=-1))
