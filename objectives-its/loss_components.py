"""Standalone loss component functions for controlled SDE training (Session 10).

Each function accepts standard tensors and returns a scalar tensor.
These were refactored out of controlled_score_training.py to enable:
  - individual gradient analysis (Step 1B)
  - unit testing each term in isolation
  - the v2 objective redesign (Step 2)

Objective version history:
  legacy (Sessions 1-9): dsm + control_weight*control_energy + path_kl_weight*path_kl
                         + quality_weight*feature_matching_loss
  v2     (Session 10+):  dsm + path_kl_weight*path_kl + quality_weight*trajectory_quality
                         + reinforce_weight*reinforce_quality + control_energy.detach()
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Legacy loss components (extracted from training loop for diagnostics)
# ---------------------------------------------------------------------------

def compute_control_energy(
    control_vec: torch.Tensor,
    step_size: float = 1.0,
) -> torch.Tensor:
    """Mean squared control norm, scaled by step_size.

    Args:
        control_vec: Control output (B, D) or (B, C, H, W) — any shape.
        step_size: SDE step size (1/T).

    Returns:
        Scalar: mean(||u||^2) * step_size
    """
    return control_vec.pow(2).mean() * step_size


def compute_path_kl(
    log_rn_terms: list[torch.Tensor],
) -> torch.Tensor:
    """Girsanov path KL from accumulated log Radon-Nikodym terms.

    KL(P^u || P^0) = E_{P^u}[log dP^u/dP^0] = +mean(sum_t log_rn_t).

    Audit 2026-07-15 (A4): this returned ``-mean(...)``.  That was only
    non-negative because the log_rn expression it consumed carried a
    compensating sign error (see sde/girsanov.py).  The sign here and the
    log_rn formula are a MATCHED PAIR -- changing either alone yields a
    "path KL" that is unbounded below in ||u||.  Callers must produce log_rn
    terms via :func:`its.sde.girsanov.girsanov_log_rn_step`.

    Args:
        log_rn_terms: List of per-step log-RN tensors, each shape (B,), as
            returned by ``girsanov_log_rn_step``.

    Returns:
        Scalar: +mean(sum_t log_rn_t), non-negative in expectation.
    """
    log_rn = torch.stack(log_rn_terms, dim=0).sum(dim=0)  # (B,)
    return log_rn.mean()


def compute_quality_loss(
    gen_samples: torch.Tensor,
    real_batch: torch.Tensor,
    feature_extractor: nn.Module,
) -> torch.Tensor:
    """Legacy feature-matching quality loss (batch statistics, low gradient signal).

    L2 distance between mean Inception feature vectors of generated and real batches.
    This is the original Session 1-9 quality term. Kept for legacy objective.

    Args:
        gen_samples: (B, C, H, W) generated images.
        real_batch: (B, C, H, W) real images from the same batch.
        feature_extractor: Frozen Inception-v3 (fc replaced by Identity).

    Returns:
        Scalar MSE loss between mean feature vectors.
    """
    def _prepare(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return x

    with torch.no_grad():
        real_feats = feature_extractor(_prepare(real_batch.detach())).mean(dim=0)
    gen_feats = feature_extractor(_prepare(gen_samples)).mean(dim=0)
    return F.mse_loss(gen_feats, real_feats)


# ---------------------------------------------------------------------------
# v2 objective: trajectory quality loss (per-sample, stronger signal)
# ---------------------------------------------------------------------------

def compute_trajectory_quality_loss(
    generated_samples: torch.Tensor,
    real_samples: torch.Tensor,
    feature_extractor: nn.Module,
    temperature: float = 1.0,
    return_per_sample: bool = False,
):
    """Per-sample trajectory quality loss using nearest-neighbor feature similarity.

    For each generated sample, finds the nearest real sample in feature space
    and computes the negative log of the exponential similarity:
        loss = mean(-log(exp(-||phi(gen_i) - phi(real_j*)||^2 / tau)))
             = mean(||phi(gen_i) - phi(real_j*)||^2) / tau

    where j* = argmin_j ||phi(gen_i) - phi(real_j)||^2.

    This provides a per-sample gradient signal — each generated image gets a
    gradient proportional to how far it is from the nearest real image in
    feature space, rather than only the batch mean.

    Args:
        generated_samples: (B, C, H, W) generated images.
        real_samples: (B, C, H, W) real images (same batch, any ordering).
        feature_extractor: Frozen feature extractor (e.g. truncated Inception).
        temperature: Softness of the similarity measure. Higher = more forgiving.

    Returns:
        Scalar non-negative quality loss, or (scalar, per_sample_dist2) if return_per_sample=True.
    """
    def _prepare(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return x

    with torch.no_grad():
        real_feats = feature_extractor(_prepare(real_samples.detach()))  # (B, D)

    gen_feats = feature_extractor(_prepare(generated_samples))  # (B, D)

    # Pairwise squared L2 distances: (B_gen, B_real)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    #
    # Audit 2026-07-15 (C1): `cross` was previously computed INSIDE a
    # torch.no_grad() block.  The forward value was bit-exact, but the backward
    # pass then saw dist2 = ||phi(gen)||^2 + const, so the gradient was
    # 2*gen_feats/(B*tau) -- pure shrinkage of the Inception features toward the
    # origin, with ZERO pull toward the nearest real sample.  The
    # nearest-neighbour search was entirely gradient-irrelevant.  Only real_sq
    # (which depends solely on the already-detached real_feats) may be detached.
    gen_sq = (gen_feats ** 2).sum(dim=1, keepdim=True)   # (B, 1)  -- carries grad
    real_sq = (real_feats ** 2).sum(dim=1, keepdim=True)  # (B, 1) -- real_feats already detached
    cross = gen_feats @ real_feats.T                        # (B_gen, B_real) -- MUST carry grad
    dist2 = gen_sq + real_sq.T - 2 * cross                     # (B_gen, B_real)
    dist2 = dist2.clamp(min=0.0)

    # Nearest-neighbor distance for each generated sample
    nn_dist2, _ = dist2.min(dim=1)  # (B_gen,)

    # -log(exp(-d^2 / tau)) = d^2 / tau
    scalar_loss = nn_dist2.mean() / temperature
    if return_per_sample:
        return scalar_loss, nn_dist2.detach() / temperature
    return scalar_loss


# ---------------------------------------------------------------------------
# v2 objective: REINFORCE-style quality loss
# ---------------------------------------------------------------------------

def compute_reinforce_quality_loss(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    advantage_clip: float = 5.0,
    log_prob_clip: float = 50.0,
) -> torch.Tensor:
    """REINFORCE policy-gradient quality loss with a standardised, bounded surrogate.

    Computes: loss = -mean(clip(A_std) * clip(log_prob)) where A_std is the
    batch-standardised advantage.

    Audit 2026-07-15 (Part B, phase 2).  The pre-audit form was

        baseline = rewards.detach().mean()
        return -((rewards.detach() - baseline) * log_probs).mean()

    which is UNBOUNDED BELOW in ||u|| and drove run 5A to total_loss = -1.59e14
    with a control/score magnitude ratio of 55,000, despite grad_clip=1.0 being
    active.  Three independent defects combined:

    1. `log_probs` (the Girsanov exponent) grows QUADRATICALLY in ||u|| and is
       unbounded.  With a mean-zero baseline, roughly half of every batch has
       A_i < 0 by construction, and for those samples -A_i*log_prob_i -> -inf as
       ||u|| -> inf.  Every batch therefore contained samples whose gradient
       drove ||u|| to infinity.  Gradient clipping cannot rescue an objective
       that is unbounded below -- it bounds the step size, not the direction, so
       clipped descent still marches to infinity at a bounded rate.  That is
       exactly the observed steady monotonic blow-up.
    2. The advantage was NOT standardised, and `rewards` are not in [0,1] as the
       old docstring claimed -- they are negated feature distances reaching ~1e6.
    3. Fixing the Girsanov sign alone does NOT fix this: with a correct
       log_rn ~ +||u||^2/(2*var), the pathology merely migrates to the
       POSITIVE-advantage samples.  The surrogate itself must be bounded.

    Standardising the advantage makes the estimator scale-free (and is the
    standard variance-reduction step); clipping both factors bounds the
    surrogate so that no finite batch can reward ||u|| -> inf.  Both clips are
    applied to the *coefficient* of the score function, so the estimator remains
    an unbiased policy gradient in the unclipped region.

    Args:
        rewards: (B,) per-trajectory quality rewards (higher = better). Any scale.
        log_probs: (B,) summed Girsanov log-RN per trajectory (see sde/girsanov.py).
        advantage_clip: Symmetric bound on the standardised advantage.
        log_prob_clip: Symmetric bound on the log-probability factor.

    Returns:
        Scalar REINFORCE loss, bounded below (minimise to improve quality).
    """
    r = rewards.detach()
    advantage = r - r.mean()
    # Standardise: makes the estimator invariant to the (arbitrary) reward scale.
    std = r.std()
    if torch.isfinite(std) and std > 1e-8:
        advantage = advantage / std
    advantage = advantage.clamp(-advantage_clip, advantage_clip)

    lp = log_probs.clamp(-log_prob_clip, log_prob_clip)
    return -(advantage * lp).mean()


# ---------------------------------------------------------------------------
# v2 objective: distillation loss for joint fine-tuning
# ---------------------------------------------------------------------------

def compute_distillation_loss(
    current_score: torch.Tensor,
    reference_score: torch.Tensor,
) -> torch.Tensor:
    """L2 distillation regularizer keeping fine-tuned score model near pretrained weights.

    Args:
        current_score: Score output from the trainable model, shape (B, C, H, W).
        reference_score: Score output from the frozen reference model, same shape.

    Returns:
        Scalar: mean((s_current - s_reference)^2).
    """
    return F.mse_loss(current_score, reference_score.detach())


# ---------------------------------------------------------------------------
# Helper: compute per-trajectory log probabilities for REINFORCE
# ---------------------------------------------------------------------------

def compute_trajectory_log_probs(
    control_vecs: list[torch.Tensor],
    noise_vecs: list[torch.Tensor],
    beta_ts: list[float],
    step_size: float,
) -> torch.Tensor:
    """Compute sum_t log p(noise_t | u_theta_t) for REINFORCE.

    Under the controlled SDE the Brownian noise at step t is:
        dW_t ~ N(0, beta_t * step_size * I)
    The log probability of the observed noise given the control drift is:
        log_prob_t = sum_t [u_t . noise_t * sqrt(beta_t * step_size)
                            - 0.5 * ||u_t||^2 * step_size]
    (This is the same as the Girsanov log-RN used for path KL.)

    Args:
        control_vecs: List of control vectors at each step, each (B, D).
        noise_vecs: List of Brownian noise samples at each step, each (B, D).
        beta_ts: List of beta(t) values at each step.
        step_size: SDE step size (1/T).

    Returns:
        (B,) tensor of sum log probabilities over trajectory steps.
    """
    log_prob_terms = []
    for u, w, beta_t in zip(control_vecs, noise_vecs, beta_ts):
        # Flatten spatial dims if needed
        u_flat = u.reshape(u.shape[0], -1)
        w_flat = w.reshape(w.shape[0], -1)
        sqrt_bs = math.sqrt(beta_t * step_size)
        term = (u_flat * w_flat).sum(dim=1) * sqrt_bs - 0.5 * u_flat.pow(2).sum(dim=1) * step_size
        log_prob_terms.append(term)
    return torch.stack(log_prob_terms, dim=0).sum(dim=0)  # (B,)


# ---------------------------------------------------------------------------
# v2 objective: structured total loss
# ---------------------------------------------------------------------------

def compute_v2_total_loss(
    dsm_loss: torch.Tensor,
    path_kl: torch.Tensor,
    trajectory_quality_loss: torch.Tensor,
    reinforce_quality_loss: torch.Tensor,
    control_energy: torch.Tensor,
    path_kl_weight: float = 0.01,
    quality_weight: float = 1.0,
    reinforce_weight: float = 0.1,
    detach_control_energy: bool = True,
    control_energy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble the v2 total loss.

    Audit 2026-07-15 (C2): control energy was previously added as a bare ``+ ce``
    with an implicit weight of 1.0, while every other term carried an explicit
    coefficient, and the configured control weight was silently ignored.  With
    ``detach_control_energy=True`` it contributed zero gradient but polluted the
    reported ``total_loss``; with it ``False`` it entered at a wrong hardcoded
    weight.  It now has an explicit ``control_energy_weight``.

    Note on ``detach_control_energy=True`` (run 5A's setting): it removes the
    only term that penalises control magnitude.  Combined with a WarmupSchedule
    that zeroes ``path_kl_weight`` AND ``reinforce_weight`` for the first epochs,
    this leaves ``||u||`` with NO regulariser whatsoever -- which is how control
    energy grew 25,000x before the REINFORCE term was even switched on.  Callers
    must ensure at least one ``||u||`` regulariser is active at every epoch; see
    :func:`assert_control_regularised`.

    Returns:
        (total_loss, component_dict) where component_dict has float values
        for logging.
    """
    ce = control_energy.detach() if detach_control_energy else control_energy
    total = (
        dsm_loss
        + path_kl_weight * path_kl
        + quality_weight * trajectory_quality_loss
        + reinforce_weight * reinforce_quality_loss
        + control_energy_weight * ce
    )
    return total, {
        "dsm_loss": dsm_loss.item(),
        "path_kl": path_kl.item(),
        "trajectory_quality": trajectory_quality_loss.item(),
        "reinforce_quality": reinforce_quality_loss.item(),
        "control_energy": control_energy.item(),
    }
