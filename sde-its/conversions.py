from __future__ import annotations

"""Score / epsilon conversion utilities.

The score model trained with denoising score matching (DSM) predicts the Stein
score of the noisy distribution:

    s_theta(x, sigma) ≈ ∇_x log p_sigma(x) = -epsilon / sigma

where ``epsilon ~ N(0, I)`` is the noise added during training.  DDPM and DDIM
update rules are expressed in terms of the *noise prediction* ``eps_theta``, so
callers must convert before plugging the score into a DDPM/DDIM formula.

    eps_theta = -score * sigma       (i.e. score_to_eps)
    score     = -eps_theta / sigma   (i.e. eps_to_score, the inverse)
"""

import torch


def score_to_eps(score: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Convert score prediction to noise (epsilon) prediction.

    Args:
        score: Score estimate s_theta(x, sigma), shape (B, ...).
        sigma: Noise level used during the forward diffusion, broadcastable
               to *score*.  May be a scalar or a tensor.

    Returns:
        eps_theta = -score * sigma, same shape as *score*.
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=score.device, dtype=score.dtype)
    return -score * sigma


def eps_to_score(eps: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Convert noise (epsilon) prediction back to score estimate.

    Args:
        eps: Noise estimate eps_theta(x, sigma), shape (B, ...).
        sigma: Noise level, broadcastable to *eps*.

    Returns:
        score = -eps / sigma, same shape as *eps*.
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=eps.device, dtype=eps.dtype)
    # Clamp sigma away from zero to avoid NaN/Inf division (audit fix CAT3-2).
    sigma = sigma.clamp(min=1e-5)
    return -eps / sigma
