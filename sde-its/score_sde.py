from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..controllers import ControlPolicy
from ..controllers.neural_control import ConvControlPolicy
from ..physics.fluctuation import jarzynski_work_estimate
from .girsanov import girsanov_log_rn_step, girsanov_path_kl


def linear_beta_schedule(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min + t * (beta_max - beta_min)


@dataclass
class ScoreSDEConfig:
    beta_min: float = 0.1
    beta_max: float = 20.0
    num_steps: int = 1000
    eps: float = 1e-3
    clamp: float = 10.0
    control_weight: float = 0.0
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    corrector_steps: int = 0
    corrector_step_size: float = 0.01
    jarzynski_clip: float = 50.0
    jarzynski_subsample: int = 1
    # Audit 2026-07-15 (A1/A2).  "ve" is the only correct choice for a model
    # trained by score_training.py, which uses the VE kernel x + sigma*noise and
    # the DSM target -noise/sigma.  "legacy_vp" preserves the pre-audit drift
    # (0.5*beta*(-x - score)) *solely* so archived runs can be reproduced; it
    # does not sample from the data distribution -- with an exact analytic score
    # it converges to ~84x the true variance and does not improve with NFE.
    # Never use it to produce a result.
    parameterisation: str = "ve"


class ScoreSDESimulator:
    def __init__(self, score_model: torch.nn.Module, config: ScoreSDEConfig) -> None:
        self.model = score_model
        self.config = config

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        # log-uniform schedule consistent with DSM training
        return self.config.sigma_min * (self.config.sigma_max / self.config.sigma_min) ** t

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        control: Optional[ControlPolicy] = None,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        step_size = 1.0 / cfg.num_steps
        ts = torch.linspace(1.0, cfg.eps, cfg.num_steps, device=device)
        # VE prior: x ~ N(0, sigma_max^2 I).  (Pre-audit this was always N(0,I),
        # which is only correct when sigma_max == 1.)
        x = torch.randn(shape, device=device) * (
            cfg.sigma_max if cfg.parameterisation == "ve" else 1.0
        )
        self.model.eval()

        # Audit A3: control_weight defaults to 0.0 and the control branch is
        # gated on it, so passing a policy while leaving control_weight unset
        # silently produced an UNCONTROLLED sample.  Six eval call-sites did
        # exactly that, which is why every "controlled" FID in the pre-audit
        # record equals the uncontrolled baseline.  Refuse to fail silently.
        if control is not None and cfg.control_weight <= 0:
            raise ValueError(
                "A control policy was supplied but control_weight="
                f"{cfg.control_weight} disables it, so the sample would be "
                "UNCONTROLLED. Pass control_weight>0 to apply the control, or "
                "control=None to sample the baseline explicitly. "
                "(Audit 2026-07-15 A3: this silent no-op voided every controlled "
                "result in the project.)"
            )
        nfe_pred = 0
        nfe_corr = 0
        control_energy_terms: list[torch.Tensor] = []
        log_rn_terms: list[torch.Tensor] = []

        for idx, t in enumerate(ts):
            t_batch = torch.full((shape[0], 1), t, device=device)
            noise = torch.randn_like(x)
            sigma_t = self._sigma(t)
            score = self.model(x, sigma=sigma_t).detach()

            if cfg.parameterisation == "ve":
                # Correct VE reverse-diffusion predictor (Song et al. 2021):
                #   dx = -(d sigma^2/dt) * score dt + sqrt(d sigma^2/dt) dW_bar
                # Discretised over the descending sigma grid, with
                #   dvar = sigma_i^2 - sigma_{i-1}^2 > 0:
                #   x <- x + dvar * score + sqrt(dvar) * z
                # This matches the VE kernel the score model was trained on
                # (score_training.py: noisy = x + sigma*noise, target = -noise/sigma).
                sigma_next = self._sigma(ts[idx + 1]) if idx + 1 < len(ts) else torch.zeros_like(sigma_t)
                dvar = (sigma_t ** 2 - sigma_next ** 2).clamp(min=0.0)
                drift = dvar * score
                step_var = dvar
            elif cfg.parameterisation == "legacy_vp":
                # Pre-audit drift, retained only for reproducing archived runs.
                beta_t = linear_beta_schedule(t, cfg.beta_min, cfg.beta_max)
                drift = 0.5 * beta_t * (-x - score) * step_size
                step_var = beta_t * step_size
            else:
                raise ValueError(
                    f"Unknown parameterisation {cfg.parameterisation!r}; expected 've' or 'legacy_vp'."
                )

            if cfg.clamp:
                drift = drift.clamp(-cfg.clamp, cfg.clamp)

            control_term = torch.zeros_like(x)
            if control is not None and cfg.control_weight > 0:
                # Support both MLP (expects flattened 1-D input) and
                # ConvControlPolicy (expects 4-D image input directly).
                if isinstance(control, ConvControlPolicy):
                    control_vec_4d = control(x, t_batch)
                    control_vec = control_vec_4d.view(x.shape[0], -1)
                else:
                    flat_x = x.view(x.shape[0], -1)
                    control_vec = control(flat_x, t_batch)
                # The drift shift actually integrated into x.  The Girsanov
                # exponent below MUST describe this exact tensor -- passing the
                # bare control (pre-audit behaviour) is inconsistent with the
                # simulated SDE whenever control_weight != 1.
                control_term = cfg.control_weight * control_vec.view_as(x) * torch.sqrt(step_var)
                control_energy_terms.append((control_vec.pow(2).mean()) * step_size)
                if cfg.jarzynski_subsample <= 1 or idx % cfg.jarzynski_subsample == 0:
                    log_rn_terms.append(
                        girsanov_log_rn_step(control_term, noise, step_var)
                    )

            x = x + drift + control_term + torch.sqrt(step_var) * noise
            nfe_pred += 1

            # Corrector: simple Langevin steps
            if cfg.corrector_steps > 0:
                for _ in range(cfg.corrector_steps):
                    score_corr = self.model(x, sigma=sigma_t).detach()
                    noise_corr = torch.randn_like(x)
                    step_corr = cfg.corrector_step_size
                    step_noise = torch.sqrt(torch.tensor(2.0 * step_corr, device=device))
                    x = x + step_corr * score_corr + step_noise * noise_corr
                    nfe_corr += 1

        if not return_stats:
            return x
        stats = {
            "nfe_predictor": float(nfe_pred),
            "nfe_corrector": float(nfe_corr),
            "nfe_total": float(nfe_pred + nfe_corr),
        }
        if control_energy_terms:
            stats["control_energy"] = float(torch.stack(control_energy_terms).sum().item())
        if log_rn_terms:
            log_rn = torch.stack(log_rn_terms, dim=0).sum(dim=0)
            if cfg.jarzynski_subsample > 1:
                # Audit C6: subsampling summed only every N-th step with no
                # rescaling, silently under-counting path_kl by ~N.
                log_rn = log_rn * float(cfg.jarzynski_subsample)
            # Sign: KL = +mean(log_rn).  See sde/girsanov.py -- this sign and the
            # log_rn formula are a matched pair and must never be changed apart.
            stats["path_kl"] = float(girsanov_path_kl(log_rn).item())
            try:
                jar = float(jarzynski_work_estimate(log_rn.cpu().numpy(), beta=1.0, clip=cfg.jarzynski_clip))
                stats["jarzynski"] = jar
                # Audit: the estimate pins to the clip rail once any trajectory
                # saturates, at which point it is meaningless.  Surface that
                # rather than reporting a rail value as a measurement.
                if cfg.jarzynski_clip and abs(abs(jar) - cfg.jarzynski_clip) < 1e-6:
                    stats["jarzynski_saturated"] = 1.0
            except Exception:
                pass
        return x, stats


# Alias for API compatibility.
ScoreSDE = ScoreSDESimulator
