from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    import torchsde  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    torchsde = None


@dataclass
class TorchSDEConfig:
    dt: float = 1e-3
    method: str = "euler"
    adaptive: bool = False
    rtol: float = 1e-3
    atol: float = 1e-4


class _ControlledSDE(torchsde.SDEIto if torchsde else object):  # type: ignore
    def __init__(self, score_model: torch.nn.Module, control: Optional[torch.nn.Module], beta_min: float, beta_max: float):
        if torchsde is None:
            raise ImportError("torchsde is not installed. Please `pip install torchsde` to use TorchSDE integration.")
        super().__init__()
        self.score_model = score_model
        self.control = control
        self.beta_min = beta_min
        self.beta_max = beta_max

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        sigma_t = torch.sqrt(t.clamp(min=1e-5))
        score = self.score_model(y, sigma=sigma_t)
        drift = 0.5 * beta_t * (-y - score)
        if self.control is not None:
            t_batch = t.expand(y.shape[0]).unsqueeze(-1)
            u = self.control(y.view(y.shape[0], -1), t_batch).view_as(y)
            drift = drift + u
        return drift

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return torch.sqrt(beta_t) * torch.ones_like(y)


def sample_with_torchsde(
    score_model: torch.nn.Module,
    control: Optional[torch.nn.Module],
    shape: Tuple[int, ...],
    beta_min: float,
    beta_max: float,
    cfg: TorchSDEConfig,
    device: torch.device,
) -> torch.Tensor:
    if torchsde is None:
        raise ImportError("torchsde is not installed. Please `pip install torchsde` to use TorchSDE integration.")
    sde = _ControlledSDE(score_model, control, beta_min, beta_max)
    y0 = torch.randn(shape, device=device)
    ts = torch.tensor([1.0, 1e-3], device=device)  # integrate from 1 -> eps
    ys = torchsde.sdeint(
        sde,
        y0,
        ts,
        dt=cfg.dt,
        method=cfg.method,
        adaptive=cfg.adaptive,
        rtol=cfg.rtol,
        atol=cfg.atol,
        logqp=False,
    )
    return ys[-1]
