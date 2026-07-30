from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    import torchcde  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torchcde = None


@dataclass
class CDEControlConfig:
    input_channels: int
    hidden_channels: int = 64
    hidden_hidden_channels: int = 128
    time_hidden_channels: int = 8


class CDEFunc(nn.Module):
    def __init__(self, hidden_channels: int, hidden_hidden_channels: int, input_channels: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear2 = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.linear1(z))
        out = self.linear2(z)
        out = out.view(z.size(0), self.hidden_channels, self.input_channels)
        return out


class CDEControlPolicy(nn.Module):
    def __init__(self, config: CDEControlConfig):
        super().__init__()
        if torchcde is None:
            raise ImportError("torchcde not installed. Please `pip install torchcde` to use CDE controllers.")
        self.config = config
        self.initial = nn.Linear(config.input_channels, config.hidden_channels)
        self.func = CDEFunc(config.hidden_channels, config.hidden_hidden_channels, config.input_channels)

    def forward(self, coeffs: torch.Tensor, eval_times: torch.Tensor) -> torch.Tensor:
        if torchcde is None:
            raise ImportError("torchcde not installed. Please `pip install torchcde` to use CDE controllers.")
        X = torchcde.LinearInterpolation(coeffs)
        z0 = self.initial(X.evaluate(X.interval[0]))
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=eval_times)
        return zt


def build_cde_control_policy(config: CDEControlConfig) -> CDEControlPolicy:
    return CDEControlPolicy(config)
