from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..sde.conversions import score_to_eps


@dataclass
class DDPMSamplerConfig:
    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    use_ddim: bool = False
    ddim_eta: float = 0.0


class DDPMSampler:
    def __init__(self, model: torch.nn.Module, config: DDPMSamplerConfig) -> None:
        self.model = model
        self.config = config
        self._prepare_schedule()

    def _prepare_schedule(self) -> None:
        cfg = self.config
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": alpha_bar,
        }

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> Tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        betas = self.register["betas"].to(device)
        alphas = self.register["alphas"].to(device)
        alpha_bar = self.register["alpha_bar"].to(device)

        x = torch.randn(shape, device=device)
        nfe = 0
        for t in reversed(range(cfg.num_steps)):
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bar[t]
            # sigma_t is the noise level for this step under the VP schedule.
            # Clamp (1 - alpha_bar_t) away from zero to avoid NaN division when
            # t approaches 0 and alpha_bar_t → 1.  (Audit CAT5, Session 5.)
            one_minus_ab = (1.0 - alpha_bar_t).clamp(min=1e-5)
            sigma_t = torch.sqrt(one_minus_ab)
            # The score model was trained with DSM to predict s_theta = -eps/sigma,
            # NOT to predict eps directly.  Convert to epsilon before the DDPM update.
            score = self.model(x, sigma=sigma_t)
            eps_theta = score_to_eps(score, sigma_t)
            nfe += 1
            if t == 0:
                x0_pred = (x - sigma_t * eps_theta) / torch.sqrt(alpha_bar_t.clamp(min=1e-5))
                x = x0_pred
                break

            alpha_bar_prev = alpha_bar[t - 1]
            if cfg.use_ddim:
                # DDIM update (eta controls stochasticity)
                eta = cfg.ddim_eta
                one_minus_ab_prev = (1 - alpha_bar_prev).clamp(min=1e-5)
                sigma = eta * torch.sqrt(one_minus_ab_prev / one_minus_ab * (1 - alpha_bar_t / alpha_bar_prev).clamp(min=0))
                x0_pred = (x - sigma_t * eps_theta) / torch.sqrt(alpha_bar_t.clamp(min=1e-5))
                direction = torch.sqrt((one_minus_ab_prev - sigma**2).clamp(min=0)) * eps_theta
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = torch.sqrt(alpha_bar_prev) * x0_pred + direction + sigma * noise
            else:
                noise = torch.randn_like(x)
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / sigma_t  # sigma_t already clamped above
                x = coef1 * (x - coef2 * eps_theta) + torch.sqrt(beta_t) * noise

        stats = {"nfe_total": float(nfe)}
        return x, stats
