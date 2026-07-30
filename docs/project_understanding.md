# ITS — Project Understanding

**Written:** 2026-07-15 (audit session)
**Sources:** every module in `src/its/`, every script in `scripts/`, every config in `configs/`,
every test in `tests/`, every document in `docs/`, every results file in `data/results/`,
every session summary and audit log (Sessions 4–10), and every training log in `logs/` and `data/logs/`.

---

## 1. The initial purpose of the project

ITS (Information Track Sampler) was conceived to **model image generation as a controlled
stochastic process**, treating sampling as a **stochastic optimal control** problem rather than as
fixed inference. A pretrained score model supplies a baseline reverse-diffusion drift; ITS augments
it with a **learned neural control drift** `u_θ(x, t)` and asks what that control buys and what it costs.

### The central controlled SDE

The framework's governing equation is the reverse-time diffusion SDE augmented with a control term:

```
dx = [ f(x, t) − g(t)² ∇ₓ log p_t(x) + g(t)·u_θ(x, t) ] dt + g(t) dW̄
     └──────── pretrained score model drift ────────┘   └ learned ┘
```

In the VP parameterisation the codebase nominally targets (`beta_min=0.1`, `beta_max=10.0`),
`f(x,t) = −½β(t)x` and `g(t) = √β(t)`, integrated backwards from `t=1` to `t≈0`.

The control is not free. Girsanov's theorem gives the exact information cost of deviating from the
uncontrolled path measure:

```
KL(P^u ‖ P⁰) = E_{P^u} [ ∫₀¹ ‖u(x,t)‖² / (2 g(t)²) dt ]
```

This is what makes the framework more than "add a residual network": every unit of control is
charged against a path-space information budget, and the resulting trade-off is measurable.

### The five quantities the framework tracks

ITS's distinguishing claim is that it accounts for all five simultaneously:

| # | Quantity | Meaning | Where computed |
|---|----------|---------|----------------|
| 1 | **Sample quality** | FID / Inception Score of generated images | `src/its/eval/evaluator.py` |
| 2 | **Control energy** | `E[‖u‖²]` — the magnitude of the applied control | `simulate_path`, `score_sde.sample` |
| 3 | **Entropy production** | Non-equilibrium thermodynamic irreversibility of the sampling path | `src/its/physics/entropy.py` |
| 4 | **Information cost** | Path-space KL via the Girsanov log Radon–Nikodym derivative | `simulate_path`, `loss_components.compute_path_kl` |
| 5 | **Compute cost** | NFE (number of function evaluations) per sample | `evaluator.py`, sampler `stats` |

The intended scientific payoff is a **Pareto frontier** over (quality, energy, information, compute),
supported by a thermodynamic analysis suite: Crooks fluctuation theorem verification, Jarzynski
equality checks, and entropy-production profiles. The thermodynamic framing is the project's
intellectual identity — image sampling as a non-equilibrium physical process with a full accounting.

---

## 2. The final goal of the project

**Success = a rigorous, publishable research paper whose every claim is supported by
scientifically sound experimental results.**

Not "a paper that says ITS wins." A paper whose claims match what the experiments actually show.

### What the realistic contribution was believed to be (entering this session)

The project's own record (Sessions 9–10) had converged on this story:

1. The controller **collapses** to near-zero output (`‖u‖/‖s‖ = 0.061%` at epoch 50, a 55× reduction
   from epoch 5) under the legacy ("v1") objective.
2. The collapse was diagnosed as **thermodynamically motivated**: path-KL and control energy are both
   minimised at `u = 0`, so collapse is the global optimum of the v1 objective, not a training failure.
3. Session 10 implemented a **redesigned "v2" objective** to fix it (detached control energy,
   trajectory quality loss, REINFORCE term, warm-up schedule, two-phase LR).
4. The planned paper: *collapse diagnosis + redesigned objective + thermodynamic analysis suite*,
   targeting TMLR or a specialised venue; a Pareto improvement would have justified a top venue.

### What this audit establishes instead

**That contribution is not currently supported, because the experimental substrate it rests on is
broken in four independent, individually-fatal ways.** See `docs/current_state_diagnosis.md` for
proofs. In brief:

- The score-SDE sampler **does not sample from the data distribution**. With an exact analytic score
  it converges to ~84× the true variance and does not improve with NFE — matching the observed flat
  FID ≈ 325 at NFE 50/100/200.
- The score model is trained with a **VE** kernel (`x + σ·ε`) but sampled with **VP** dynamics —
  incompatible diffusion families.
- **Control was silently disabled in every evaluation** (`control_weight` defaults to `0.0` and is
  never passed), so every "controlled" FID is the uncontrolled sampler's FID.
- The **Girsanov/path-KL formula is wrong** by a factor of β_t — and path-KL is one of the five
  headline quantities.

The "FID identical to baseline ⇒ collapse" inference is therefore invalid: those FIDs are identical
for a trivial reason (they are the same computation). The collapse *ratio* remains a real measurement
of training dynamics, but it was measured inside dynamics that integrate the wrong SDE.

### The honest restatement of the goal

The final goal is unchanged in spirit but the path to it is longer than the project believed:

> A paper that demonstrates and analyses image generation as a controlled stochastic process with
> full thermodynamic and information-theoretic accounting — **on a sampler that provably samples from
> the data distribution, with control provably applied, and with a path-KL that is provably the
> Girsanov KL.**

Every one of those three "provably"s currently fails and must be established, with regression tests,
before any result can support any claim. Whether the eventual contribution is a Pareto improvement,
an honest negative result, or a methodological/diagnostic contribution **cannot be known until the
substrate is correct and the experiments are re-run.** Any framing chosen now would be a guess.

What this audit *does* contribute, and what survives independently of the re-runs, is a rigorous
**failure analysis**: four proven defects, each with a minimal reproduction, plus the observation
that a 77-test suite was fully green throughout — because it tested shapes, finiteness and imports,
never scientific correctness. That is a real and publishable lesson about validating scientific
software, but it is a lesson about *method*, not about controlled diffusion.
