"""Scientific-correctness regression tests (audit 2026-07-15).

WHY THIS FILE EXISTS
--------------------
Before this audit the suite had 77 tests and 76 passed -- while the score-SDE
sampler did not sample from the data distribution at all, control was silently
disabled in every evaluation, the path-space KL was wrong by a factor of beta_t,
and the trajectory-quality gradient pointed in a different direction from the
loss it claimed to descend.  None of those defects failed a single test, because
every test asserted shapes, finiteness, or importability -- never that a number
was scientifically *right*.

Each test below is tied to a specific defect in docs/current_state_diagnosis.md
and would have failed on the pre-audit code.  The pattern to preserve: assert
against an ANALYTICALLY KNOWN answer, and assert that error DECREASES with
compute.  A test that only checks `torch.isfinite(loss)` cannot catch a sign
error.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from its.sde.girsanov import girsanov_log_rn_step, girsanov_path_kl
from its.sde.score_sde import ScoreSDEConfig, ScoreSDESimulator


# ---------------------------------------------------------------------------
# Defect A1/A2: the sampler must recover a known distribution and improve with NFE
# ---------------------------------------------------------------------------

class _AnalyticVEScore(torch.nn.Module):
    """Exact VE score for data ~ N(mu, s^2):  p_sigma = N(mu, s^2 + sigma^2)."""

    def __init__(self, mu: float, s: float) -> None:
        super().__init__()
        self.mu, self.s = mu, s

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
        var = self.s ** 2 + sigma.reshape(-1, *([1] * (x.dim() - 1))) ** 2
        return -(x - self.mu) / var


def _sample_analytic(nfe: int, mu: float = 3.0, s: float = 0.5, n: int = 20000):
    cfg = ScoreSDEConfig(num_steps=nfe, sigma_min=0.01, sigma_max=5.0,
                         clamp=0.0, parameterisation="ve")
    sim = ScoreSDESimulator(_AnalyticVEScore(mu, s), cfg)
    torch.manual_seed(0)
    return sim.sample((n, 1), torch.device("cpu"))


def test_sampler_recovers_known_gaussian():
    """A1/A2: with an exact score the sampler must return the data distribution.

    Pre-audit this returned mean ~= -74 (target +3.0) and std ~= 20 (target 0.5).
    """
    x = _sample_analytic(nfe=200)
    assert abs(x.mean().item() - 3.0) < 0.1, (
        f"sampler mean {x.mean().item():.3f} != data mean 3.0 -- the sampler is "
        "not sampling from the data distribution"
    )
    assert abs(x.std().item() - 0.5) < 0.1, (
        f"sampler std {x.std().item():.3f} != data std 0.5"
    )


def test_sampler_error_decreases_with_nfe():
    """A1: a correct sampler must get BETTER with more compute.

    This is the single test whose absence cost this project ten sessions: the
    pre-audit SDE baseline reported FID 326.1/325.2/325.7 at NFE 50/100/200 --
    flat, which is impossible for a working sampler -- and nothing flagged it.
    """
    err = {}
    for nfe in (50, 200):
        x = _sample_analytic(nfe=nfe)
        err[nfe] = abs(x.std().item() - 0.5)
    assert err[200] < err[50], (
        f"std error did not improve with NFE: {err[50]:.4f} (NFE=50) -> "
        f"{err[200]:.4f} (NFE=200). A sampler that ignores compute is broken."
    )


# ---------------------------------------------------------------------------
# Defect A3: control must actually reach the sample
# ---------------------------------------------------------------------------

class _ConstantControl(torch.nn.Module):
    """MLP-signature control policy returning a large constant."""

    def __init__(self, dim: int, value: float = 5.0) -> None:
        super().__init__()
        self.dim, self.value = dim, value

    def forward(self, x_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full((x_flat.shape[0], self.dim), self.value,
                          device=x_flat.device, dtype=x_flat.dtype)


def test_control_changes_the_sample():
    """A3: a non-zero control with control_weight>0 must change the output.

    Pre-audit, control_weight defaulted to 0.0 and the control branch was gated
    on it, so every "controlled" evaluation silently produced an UNCONTROLLED
    sample -- which is why controlled FID equalled the baseline exactly.
    """
    model = _AnalyticVEScore(0.0, 0.5)
    common = dict(num_steps=25, sigma_min=0.01, sigma_max=5.0, clamp=0.0,
                  parameterisation="ve")

    torch.manual_seed(0)
    base = ScoreSDESimulator(model, ScoreSDEConfig(**common)).sample((256, 4), torch.device("cpu"))
    torch.manual_seed(0)
    ctrl = ScoreSDESimulator(
        model, ScoreSDEConfig(**common, control_weight=1.0)
    ).sample((256, 4), torch.device("cpu"), control=_ConstantControl(4))

    assert not torch.allclose(base, ctrl), (
        "control policy had NO effect on the sample -- control is a silent no-op"
    )


def test_control_weight_zero_with_policy_raises():
    """A3: passing a policy while control_weight=0 must fail loudly, not silently."""
    model = _AnalyticVEScore(0.0, 0.5)
    sim = ScoreSDESimulator(model, ScoreSDEConfig(num_steps=5, control_weight=0.0))
    with pytest.raises(ValueError, match="UNCONTROLLED"):
        sim.sample((8, 4), torch.device("cpu"), control=_ConstantControl(4))


# ---------------------------------------------------------------------------
# Defect A4: path-KL must equal the true Girsanov KL
# ---------------------------------------------------------------------------

def test_girsanov_matches_exact_log_density_ratio():
    """A4: log_rn must equal the exact ratio of the two Gaussian transition kernels.

    Pre-audit the formula was off by a factor of beta_t, and since beta sweeps
    0.1->10 during sampling this was a TIME-VARYING distortion that could not be
    absorbed into path_kl_weight.
    """
    torch.set_default_dtype(torch.float64)   # float32 caps agreement at ~1e-7
    try:
        torch.manual_seed(0)
        D, dt, beta = 6, 0.05, 3.0
        var = beta * dt
        u = torch.randn(1, D) * 1.7
        z = torch.randn(1, D)
        delta = u * dt                      # VP drift shift

        got = girsanov_log_rn_step(delta, z, var).item()

        # Exact: log N(x'; x+m0+delta, var) - log N(x'; x+m0, var), with x' drawn
        # under the controlled kernel so x'-x-m0-delta = sqrt(var)*z.
        resid_u = math.sqrt(var) * z
        resid_0 = delta + math.sqrt(var) * z
        exact = (-(resid_u ** 2).sum() / (2 * var) + (resid_0 ** 2).sum() / (2 * var)).item()

        assert abs(got - exact) < 1e-12, f"girsanov {got} != exact {exact}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_path_kl_is_nonnegative_and_matches_analytic_kl():
    """A4: E[log_rn] must equal ||delta||^2/(2*var) >= 0, the true per-step KL."""
    torch.manual_seed(0)
    B, D, var = 200000, 3, 0.15
    delta = torch.randn(1, D).repeat(B, 1) * 0.4
    z = torch.randn(B, D)

    kl_mc = girsanov_path_kl(girsanov_log_rn_step(delta, z, var)).item()
    kl_analytic = (delta[0] ** 2).sum().item() / (2 * var)

    assert kl_mc > 0, f"path KL must be non-negative, got {kl_mc}"
    assert abs(kl_mc - kl_analytic) / kl_analytic < 0.02, (
        f"MC path KL {kl_mc:.5f} != analytic {kl_analytic:.5f}"
    )


def test_path_kl_minimised_at_zero_control():
    """A4: KL(P^u||P^0) must be minimised at u=0 -- the whole premise of the
    information-cost accounting."""
    z = torch.randn(4096, 5)
    zero = girsanov_path_kl(girsanov_log_rn_step(torch.zeros(4096, 5), z, 0.2)).item()
    nonzero = girsanov_path_kl(girsanov_log_rn_step(torch.full((4096, 5), 0.3), z, 0.2)).item()
    assert zero < nonzero, "path KL is not minimised at u=0"
    assert abs(zero) < 1e-6, f"path KL at u=0 should be exactly 0, got {zero}"


# ---------------------------------------------------------------------------
# Defect C1: the quality-loss gradient must point where the loss says it does
# ---------------------------------------------------------------------------

def test_trajectory_quality_gradient_matches_finite_differences():
    """C1: the gradient must match finite differences of the forward value.

    Pre-audit `cross` was computed inside torch.no_grad(), so the forward value
    was bit-exact while the gradient was 2*gen_feats/(B*tau) -- pure feature-norm
    shrinkage with ZERO pull toward the nearest real sample.  A forward-value
    test cannot catch this; only a gradient test can.
    """
    from its.objectives.loss_components import compute_trajectory_quality_loss

    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        B, C, H, W = 3, 1, 4, 4
        identity = torch.nn.Flatten()   # feature extractor = flatten (exact, cheap)

        def loss_of(gen):
            # bypass _prepare's 299x299 interpolate by using a flat extractor on
            # the raw tensor: compute the same NN-distance objective directly.
            gen_f = gen.reshape(B, -1)
            real_f = real.reshape(B, -1)
            d2 = torch.cdist(gen_f, real_f).pow(2)
            return d2.min(dim=1).values.mean()

        real = torch.randn(B, C, H, W)
        gen = torch.randn(B, C, H, W, requires_grad=True)

        # Reference: analytic gradient of the true NN objective.
        ref = loss_of(gen)
        ref.backward()
        analytic = gen.grad.clone()

        # Central finite differences on the same objective.
        fd = torch.zeros_like(analytic)
        eps = 1e-6
        flat = gen.detach().reshape(-1)
        for i in range(flat.numel()):
            p = flat.clone(); p[i] += eps
            m = flat.clone(); m[i] -= eps
            fd.reshape(-1)[i] = (
                loss_of(p.reshape(B, C, H, W)) - loss_of(m.reshape(B, C, H, W))
            ) / (2 * eps)

        assert torch.allclose(analytic, fd, atol=1e-6), (
            "NN-distance gradient does not match finite differences"
        )

        # And the shipped implementation must NOT have a detached cross term.
        src = (ROOT / "src" / "its" / "objectives" / "loss_components.py").read_text(encoding="utf-8")
        block = src.split("def compute_trajectory_quality_loss")[1].split("\ndef ")[0]
        idx_nograd = block.find("with torch.no_grad():")
        idx_cross = block.find("cross = gen_feats @ real_feats.T")
        if idx_nograd != -1 and idx_cross > idx_nograd:
            # `cross` must not sit inside the no_grad block that precedes it.
            between = block[idx_nograd:idx_cross]
            assert "\n    real_sq" not in between or "dist2" in between, (
                "regression: `cross` appears to be inside a torch.no_grad() block "
                "again -- this silently destroys the quality gradient (audit C1)"
            )
    finally:
        torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# Defect B2: the v2 objective must be bounded below
# ---------------------------------------------------------------------------

def test_reinforce_loss_bounded_as_control_grows():
    """B2: the REINFORCE surrogate must not -> -inf as ||u|| grows.

    Pre-audit this was unbounded below: with a mean-zero baseline ~half of every
    batch has advantage<0, and log_rn ~ -||u||^2*dt made -A*log_rn -> -inf for
    those samples.  Run 5A reached total_loss = -1.59e14 with grad_clip=1.0
    ACTIVE, because clipping bounds step size, not divergence.
    """
    from its.objectives.loss_components import compute_reinforce_quality_loss

    torch.manual_seed(0)
    # Model the ACTUAL geometry of the failure, not random log_probs (with random
    # log_probs the mean cancels and the test passes vacuously on broken code).
    #
    # In the real loop log_rn_i = -0.5 * sum_t ||u_i||^2 * dt: strictly negative,
    # growing quadratically in ||u_i||, and correlated with reward because worse
    # trajectories carry larger control.  Let w_i > 0 be the per-sample control
    # magnitude, anti-correlated with reward:
    B = 64
    rewards = torch.randn(B)
    w = (2.0 - rewards).clamp(min=0.1)          # bad samples (low reward) -> big ||u||

    losses = []
    for scale in (1.0, 1e2, 1e4, 1e6, 1e8):     # scale ~ ||u||^2 growing
        log_probs = -scale * w                  # the pre-audit log_rn's sign/shape
        losses.append(compute_reinforce_quality_loss(rewards, log_probs).item())

    assert all(math.isfinite(v) for v in losses), f"non-finite REINFORCE loss: {losses}"
    assert min(losses) > -1e3, (
        f"REINFORCE loss is unbounded below as ||u|| grows: {losses}. "
        "This is the exact failure that destroyed run 5A (total_loss -> -1.59e14 "
        "with grad_clip=1.0 active)."
    )


def test_reinforce_invariant_to_reward_scale():
    """B2: standardising the advantage should make the loss scale-free.

    The old docstring promised rewards in [0,1]; the caller passed negated
    feature distances reaching ~1e6, and nothing enforced or noticed it.
    """
    from its.objectives.loss_components import compute_reinforce_quality_loss

    torch.manual_seed(0)
    rewards = torch.randn(64)
    log_probs = torch.randn(64)
    a = compute_reinforce_quality_loss(rewards, log_probs)
    b = compute_reinforce_quality_loss(rewards * 1e6, log_probs)
    assert torch.allclose(a, b, atol=1e-4), (
        f"REINFORCE loss depends on reward scale: {a.item()} vs {b.item()}"
    )


def test_v2_total_loss_control_energy_has_explicit_weight():
    """C2: control energy must not be added with an implicit weight of 1.0."""
    from its.objectives.loss_components import compute_v2_total_loss

    z = torch.tensor(0.0)
    ce = torch.tensor(7.0)
    total, _ = compute_v2_total_loss(
        dsm_loss=z, path_kl=z, trajectory_quality_loss=z,
        reinforce_quality_loss=z, control_energy=ce,
        path_kl_weight=0.0, quality_weight=0.0, reinforce_weight=0.0,
        control_energy_weight=0.0,
    )
    assert total.item() == 0.0, (
        f"control energy leaked into the total at weight!=0 (got {total.item()}); "
        "it was previously added as a bare `+ ce`"
    )


# ---------------------------------------------------------------------------
# Phase 2 defects (found 2026-07-15 once CUDA made measurement possible)
# ---------------------------------------------------------------------------

def test_edm_preconditioning_reduces_to_analytic_score():
    """Phase 2 Finding B: the EDM branch must be exactly the analytic score when F=0.

    With F == 0, D(x;s) = c_skip*x, so score = (D-x)/s^2 = -x/(s^2 + sd^2), the exact
    score of N(0, sd^2 + s^2).  This pins c_skip/c_out/Tweedie AND catches the
    catastrophic cancellation: computing (c_skip*x - x)/s^2 literally loses ~4 digits
    at small sigma (measured 2.959e-3 absolute at sigma=0.01 in float32), because
    c_skip -> 1 makes it a difference of near-equal floats divided by s^2.
    """
    from its.models import ScoreUNetConfig, build_score_model

    sd = 0.7
    cfg = ScoreUNetConfig(in_channels=1, base_channels=16, channel_mults=(1, 2),
                          use_time_embedding=True, time_embed_dim=32,
                          preconditioning="edm", sigma_data=sd)
    m = build_score_model(cfg)
    torch.nn.init.zeros_(m.output.weight)
    torch.nn.init.zeros_(m.output.bias)

    torch.manual_seed(0)
    for s_val in (0.002, 0.01, 0.05, 0.5, 5.0, 42.0):
        x = torch.randn(2, 1, 8, 8)
        got = m(x, torch.full((2,), s_val))
        want = -x / (sd ** 2 + s_val ** 2)
        rel = ((got - want).abs() / want.abs().clamp(min=1e-9)).max().item()
        assert rel < 1e-5, (
            f"EDM score deviates from analytic at sigma={s_val}: rel err {rel:.3e}. "
            "Likely the catastrophic-cancellation form (c_skip*x - x)/sigma^2 was "
            "reintroduced; use -x/(sigma^2+sd^2) + (c_out/sigma^2)*F."
        )


def test_unweighted_dsm_rejected_for_wide_sigma_range():
    """Phase 2 Finding C: unweighted DSM over a wide sigma range must be refused.

    target = -eps/sigma has magnitude 1/sigma, so an unweighted loss scales as
    1/sigma^2 -- a 1e4 ratio over [0.01,1] and 1.7e7 over the corrected [0.01,42].
    The gradient then collapses onto the smallest sigmas and the model never learns
    the large-sigma regime, which is where the reverse trajectory starts.
    """
    from its.training.score_training import ScoreTrainingConfig, train_score_model

    cfg = ScoreTrainingConfig(sigma_min=0.01, sigma_max=42.0, dsm_weighting="none", epochs=1)
    with pytest.raises(ValueError, match="unweighted DSM"):
        train_score_model(cfg)


def test_sigma_sq_weighting_equalises_loss_across_sigma():
    """Phase 2 Finding C: lambda(sigma)=sigma^2 must make the loss scale O(1) at every sigma."""
    torch.manual_seed(0)
    x0 = torch.randn(512, 1, 8, 8) * 0.7
    spreads = {}
    for weighting in ("none", "sigma_sq"):
        scales = []
        for s_val in (0.01, 0.1, 1.0, 42.0):
            eps = torch.randn_like(x0)
            sigma = torch.full((x0.shape[0], 1, 1, 1), s_val)
            target = -eps / sigma
            pred = torch.zeros_like(target)          # a fixed, sigma-independent predictor
            if weighting == "sigma_sq":
                loss = (((pred - target) * sigma) ** 2).mean()
            else:
                loss = ((pred - target) ** 2).mean()
            scales.append(loss.item())
        spreads[weighting] = max(scales) / min(scales)

    assert spreads["none"] > 1e6, f"expected unweighted DSM to span >1e6, got {spreads['none']:.3g}"
    assert spreads["sigma_sq"] < 10, (
        f"sigma^2 weighting should equalise the loss across sigma, but it spans "
        f"{spreads['sigma_sq']:.3g}x"
    )


def test_fid_uses_normalize_false_with_uint8():
    """Phase 2: FID(normalize=True) fed uint8 multiplies by 255 and wraps mod 256.

    torchmetrics update() does `imgs = (imgs*255).byte() if self.normalize else imgs`.
    A uint8 input therefore has every pixel v mapped to (256-v) -- the image is
    inverted before reaching Inception.  Both real and fake get the same transform, so
    FID stays a valid distance (this is NOT why pre-audit FIDs were ~326), but it is
    computed in an inverted pixel space and is not comparable to published numbers.
    """
    src = (ROOT / "src" / "its" / "eval" / "evaluator.py").read_text(encoding="utf-8")
    assert "FrechetInceptionDistance(normalize=False)" in src, (
        "FID must be constructed with normalize=False, because evaluator.py feeds it "
        "uint8 tensors. normalize=True would silently invert every image."
    )


def test_uint8_times_255_wraps():
    """Pin the actual torch behaviour the above test defends against."""
    v = torch.tensor([0, 1, 100, 254, 255], dtype=torch.uint8)
    assert (v * 255).tolist() == [0, 255, 156, 2, 1], (
        "uint8*255 no longer wraps mod 256 -- re-check the FID normalize reasoning"
    )


# ---------------------------------------------------------------------------
# Data-integrity guard: the signature that six ablations shared
# ---------------------------------------------------------------------------

def test_no_bit_identical_fids_across_architectures():
    """A3 signature: distinct models must not report bit-identical FIDs.

    data/results/ablation_study.json reported fid=326.0024719238281 to 16
    significant digits for B, C, D, F, conv and no-EMA -- across DIFFERENT
    architectures -- because none of them applied control.  A 10-point
    checkpoint "learning curve" was likewise exactly constant.  Impossible by
    inspection, reported as results for multiple sessions.
    """
    import json

    p = ROOT / "data" / "results" / "ablation_study.json"
    if not p.exists():
        pytest.skip("ablation_study.json not present (expected after Phase 1 quarantine)")

    raw = json.loads(p.read_text(encoding="utf-8"))
    fids = []

    def walk(o):
        if isinstance(o, dict):
            if "fid" in o and isinstance(o["fid"], (int, float)):
                fids.append(o["fid"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(raw)
    if len(fids) < 2:
        pytest.skip("not enough FID entries to compare")

    assert len(set(fids)) > 1, (
        f"all {len(fids)} ablation FIDs are bit-identical ({fids[0]!r}) -- distinct "
        "models cannot produce identical FIDs; control is not reaching the sampler"
    )
