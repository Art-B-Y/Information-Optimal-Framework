from __future__ import annotations

"""Exact discrete Girsanov log Radon-Nikodym derivative for controlled Euler steps.

This module exists to hold the *single* correct implementation of the path-space
information cost.  Prior to the 2026-07-15 audit the expression was duplicated
verbatim in three places, and all three copies were wrong (see
``docs/current_state_diagnosis.md`` A4).  Every caller must use this function.

Derivation
----------
Consider one Euler-Maruyama step in which the control shifts the drift by
``delta`` and both the controlled and uncontrolled kernels share the same noise
variance ``var``:

    controlled:    x' = x + m_0 + delta + sqrt(var) * z        (z ~ N(0, I))
    uncontrolled:  x' = x + m_0         + sqrt(var) * z'

Both transition kernels are Gaussian with identical covariance ``var * I``, so
their normalisers cancel and, evaluated on a path sampled from the *controlled*
measure P^u,

    log dP^u/dP^0 = [ -||x'-x-m_0-delta||^2 + ||x'-x-m_0||^2 ] / (2*var)
                  = [ -||sqrt(var) z||^2 + ||delta + sqrt(var) z||^2 ] / (2*var)
                  = <delta, z> / sqrt(var)  +  ||delta||^2 / (2*var)

Taking the expectation under P^u (where z is independent of delta at this step)
kills the linear term and leaves the per-step KL:

    E[log dP^u/dP^0] = ||delta||^2 / (2*var) = KL(P^u || P^0) >= 0

which is the correct, non-negative information cost, minimised at delta = 0.

Both terms carry a PLUS sign.  The pre-audit code used
``<u,z>*sqrt(beta*dt) - 0.5*||u||^2*dt`` -- a sign-inconsistent hybrid that is
not log(dP^u/dP^0) for any scaling of u unless beta == 1, and whose negated mean
only *looked* like a KL because two sign errors cancelled.

This form is parameterisation-agnostic: it is correct for VP (``delta = u*dt``,
``var = beta*dt``), for VE (``delta = u*h``, ``var = sigma_i^2 - sigma_{i-1}^2``),
and for any other Euler scheme, because it is stated in terms of the realised
drift shift and the realised noise variance rather than a specific schedule.
"""

import torch

__all__ = ["girsanov_log_rn_step", "girsanov_path_kl"]

_VAR_FLOOR = 1e-12


def girsanov_log_rn_step(
    delta: torch.Tensor,
    noise: torch.Tensor,
    var: torch.Tensor | float,
) -> torch.Tensor:
    """Per-sample log Radon-Nikodym increment for one controlled Euler step.

    Args:
        delta: (B, ...) drift shift *actually added to the state* by the control
            on this step -- i.e. exactly the tensor that was integrated into x,
            including any control_weight scaling and the dt/step factor.  Passing
            the bare control output here is a bug: the Girsanov exponent must
            describe the SDE that was simulated.
        noise: (B, ...) the standard-normal draw z used on this step (NOT the
            scaled increment sqrt(var)*z).  Must be the same tensor used to
            advance the state, or the estimator is inconsistent.
        var: Scalar or broadcastable noise variance of the step (e.g. beta*dt for
            VP, sigma_i^2 - sigma_{i-1}^2 for VE).  Clamped away from zero.

    Returns:
        (B,) per-sample log dP^u/dP^0 increment,
        ``<delta, z>/sqrt(var) + ||delta||^2/(2*var)``.
    """
    if not torch.is_tensor(var):
        var = torch.tensor(var, device=delta.device, dtype=delta.dtype)
    var = var.clamp(min=_VAR_FLOOR)

    b = delta.shape[0]
    d_flat = delta.reshape(b, -1)
    z_flat = noise.reshape(b, -1)

    linear = (d_flat * z_flat).sum(dim=1) / torch.sqrt(var).reshape(-1)
    quad = d_flat.pow(2).sum(dim=1) / (2.0 * var.reshape(-1))
    return linear + quad


def girsanov_path_kl(log_rn: torch.Tensor) -> torch.Tensor:
    """Path-space KL(P^u || P^0) from summed per-sample log-RN increments.

    Note the sign: KL = E_{P^u}[log dP^u/dP^0] = +mean(log_rn).  The pre-audit
    code returned ``-log_rn.mean()``, which was only non-negative because the
    log_rn expression itself carried a compensating sign error.  Negating a
    *correct* log_rn yields a quantity that is unbounded below in ||u|| -- so
    this sign and :func:`girsanov_log_rn_step` must always be changed together.

    Args:
        log_rn: (B,) per-sample sum of :func:`girsanov_log_rn_step` over steps.

    Returns:
        Scalar path-space KL estimate (non-negative in expectation).
    """
    return log_rn.mean()
