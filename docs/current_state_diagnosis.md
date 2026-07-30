# ITS — Current State Diagnosis

**Written:** 2026-07-15 (audit session)
**Method:** every JSONL training log and every evaluation JSON read; every checkpoint dir enumerated;
every claim cross-checked against the file on disk. Each headline finding below has a **minimal
reproduction** that was actually executed, not an argument.

> **Bottom line.** The codebase's *surface* hygiene is good — all 49 `torch.load` calls pass
> `map_location`, every division by σ / √(1−ᾱ) / β is clamped, the control reshape chain is correct,
> the DDPM/DDIM ε-conversion is correctly placed, and 76/77 tests pass. **The defects are all
> semantic**, and they are fatal. The test suite is green *because it never tests scientific
> correctness* — only shapes, finiteness, and importability.
>
> **Every experimental result in this project is void.** Not "imprecise" — void, for four
> independent reasons, each individually sufficient.

---

## Part A — The four fatal defects (each proven)

### A1. The score-SDE sampler does not sample from the data distribution
`src/its/sde/score_sde.py:66` and (duplicated verbatim) `src/its/training/controlled_score_training.py:531`

```python
drift = 0.5 * beta_t * (-x - score)          # = -½βx - ½β·score
x = x + drift * step_size + sqrt(beta_t*step_size)*noise   # step_size = +1/N, while t DESCENDS
```

The reverse-time VP SDE, integrated with a **positive** step while `t` walks from 1→0, requires
`drift = +½βx + β·score`. The code has **the sign inverted on both terms** and **a factor of 2 on the
score** (½β instead of β). It is not a reverse-time sampler; it is a noising process.

**Proof (executed, exact analytic score, data `N(0, 0.5²)`):**

| steps | code's rule → variance | corrected rule → variance |
|-------|------------------------|---------------------------|
| 50    | 20.32                  | 0.2570 |
| 100   | 20.65                  | 0.2525 |
| 200   | 21.00                  | 0.2511 |
| 500   | 21.13                  | 0.2516 |

Target variance is **0.25**. The corrected rule converges to it and improves with steps. The code's
rule converges to **≈84× the true variance and does not improve with NFE**.

**This exactly predicts the observed data:** `baseline_evals_fashionmnist.json` reports SDE FID
**326.09 / 325.20 / 325.69** at NFE **50 / 100 / 200** — flat, no improvement with compute. A working
sampler must improve with NFE. An independent reader agent reproduced this with a `N(3.0, 0.5²)`
target: the loop returns mean ≈ −9.87 (target +3.0), and with the score forced to zero it returns
`N(0,1)` — i.e. **the loop relaxes to noise regardless of the score model, and the score term actively
pushes away from the data.**

### A2. The score model is VE-trained but VP-sampled (incompatible kernels)
`src/its/training/score_training.py:242-248` vs `src/its/sde/score_sde.py:62-66`

```python
# TRAINING — Variance Exploding:
sigma = _sample_sigma(x, 0.01, 1.0)   # log-uniform
noisy = x + sigma * noise             # VE kernel: no √ᾱ scaling
target = -noise / sigma               # correct VE score
```
```python
# SAMPLING — Variance Preserving:
sigma_t = sigma_min*(sigma_max/sigma_min)**t     # VE geometric σ
beta_t  = beta_min + t*(beta_max-beta_min)       # VP linear β
drift   = 0.5*beta_t*(-x - score)                # VP dynamics
```

There is no SDE for which this (σ-schedule, β-schedule) pair is a discretisation. The model learns
`∇log p_σ` for the VE perturbation; the sampler applies VP drift. Additionally `sigma_max=1.0` is far
too small for a VE prior, so the `x = randn(...)` initialisation does not match `p_σmax` either.

This also explains why the *best* baseline is poor: DDPM (a separate, algebraically-correct sampler)
reaches FID **197** at NFE=200 — but a healthy FashionMNIST diffusion model reaches **FID < 20**.
DDPM "works" only because it feeds the model `σ_t = √(1−ᾱ_t) ∈ [0,1]`, which happens to overlap the
VE training range — a coincidence, not a correctness argument.

### A3. Control was silently disabled in **every** evaluation
`src/its/sde/score_sde.py:24` (`control_weight: float = 0.0`) and `:70` (the gate)

```python
if control is not None and cfg.control_weight > 0:      # never true at eval
```
```python
# scripts/eval_controlled_multiseed.py:125 — control_weight NOT passed → 0.0
sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=args.nfe, sigma_min=0.01, sigma_max=1.0)
# :51 — eval_one REBUILDS the config, copying 5 fields and DROPPING control_weight → 0.0
nfe_cfg = ScoreSDEConfig(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max,
                         num_steps=nfe, sigma_min=sde_cfg.sigma_min, sigma_max=sde_cfg.sigma_max)
```

The control policy is loaded, passed to `evaluate_sampler`, and **never applied**. Six eval call-sites
share this defect.

**This single bug explains the project's entire "collapse" evidence base:**

| Observation | Real explanation |
|---|---|
| Controlled FID 327.47 ≈ SDE baseline 325–326 | They are **the same computation** |
| `ablation_study.json`: B, C, D, F, conv, no-EMA all FID `326.0024719238281` to 16 digits — across *different architectures* | None applied control |
| `conv_checkpoint_fid_trajectory.json`: all 10 epochs (5→50) FID `335.1768493652344` identical | Checkpoint weights never reach the sampler |
| `controlled_config_d_multiseed.json` byte-identical to `controlled_conv_multiseed.json` | Both are the uncontrolled sampler at the same seeds |

> **Consequence for the paper.** The inference *"controlled FID = baseline FID ⇒ the controller
> collapsed"* is **invalid**. Those FIDs are identical for a trivial reason. The collapse *ratio*
> (`‖u‖/‖s‖ = 0.061%`) is a genuine measurement — it comes from training diagnostics, which do apply
> control — but the FID half of the argument evaporates, and it was measured inside the A1 dynamics.

### A4. The Girsanov log-Radon–Nikodym / path-KL formula is wrong
`controlled_score_training.py:544`, `sde/score_sde.py:83`, `objectives/loss_components.py:238` (three copies)

```python
log_rn_step = (control_vec*noise_flat).sum(1)*sqrt(beta_t*step_size) - 0.5*control_vec.pow(2).sum(1)*step_size
```

The correct discrete Girsanov exponent for `x' = x + (b+u)dt + √(βdt)·ε`, evaluated under `P^u`, is

```
log_rn_true = ⟨u,ε⟩·√(dt/β) + ½‖u‖²·dt/β
```

**Proof (executed, against exact Gaussian log-densities):** the correct formula matches ground truth
to **3.3e-16**; the code's formula is off by **0.94 in a single step**, and the code's `path_kl` is off
by exactly a factor of **β_t = 3.0** in the test. Independently Monte-Carlo verified to **4.4e-15**.

Since β sweeps **0.1 → 10** during sampling, this is a **time-varying** distortion — it cannot be
absorbed into `path_kl_weight`. Path-space KL is one of the five headline quantities the framework
claims to track; **every path-KL number ever reported by this project is wrong.**

Two precise corrections from adversarial verification, both of which matter for the fix:
- The code's quadratic term contains **no β at all** (it does not "multiply by β" there — it *omits*
  the required `1/β`). The net error is a factor of β in both terms, but by a different mechanism.
- The error is **sign-compensated** by `path_kl = -log_rn.mean()` (line 549), which makes the reported
  path-KL *positive* and roughly `½‖u‖²dt` — plausible-looking, which is why it survived ten sessions.
  **Fixing line 544 alone would flip path-KL negative and unbounded below.** The correct fix is
  joint: new formula **and** `path_kl = +log_rn.mean()`.

---

## Part B — Status of the Session 10 redesigned objective

**Question posed by the brief:** was the controller-activation run (v2 objective) completed and
evaluated? **Answer: it was completed to 39/100 epochs, it diverged catastrophically, and it was
never evaluated.** It is not a pending run — it is a *failed* run, and the reason it failed is a bug.

### The run
- Command: `python scripts/train_redesigned_v2.py --run 5a`, `grad_clip=1.0` **active**
- Log: `data/logs/controlled_v2_5a_seed42.jsonl` (764 lines)
- Reached **epoch 39 of 100**; **176.1 wall-clock hours** (vs. the "10–15 h" estimate — see Part D)
- Checkpoints: epochs 10, 20, 30 + `controlled_last.pt`. **Never evaluated** (no v2 results JSON exists)
- Runs **5B and 5C were never started**: `checkpoints/controlled_v2_5b_joint/` and
  `controlled_v2_5c_nopkL/` are **empty**

### The divergence (controller output magnitude ratio `‖u‖/‖s‖`)

| epoch | 1 | 5 | 10 | 20 | 30 | 39 |
|---|---|---|---|---|---|---|
| ratio | 19.9 | 139.6 | 937.7 | 1.96e4 | 5.38e4 | 3.46e4 |
| total_loss | 3607 | — | — | — | −6.9e14 | **−1.59e14** |
| grad_norm_control | 1.6e4 | — | — | — | `inf` | 2.06e16 |

- `total_loss` first went **negative at epoch 6**; `grad_norm_control` exceeded 1e3 **at epoch 1**
- `path_kl` reached **7.3e10**; `control_energy` **1.87e8**; `quality_loss` grew **3297 → 2.5e6**
  (samples got *worse*); `jarzynski` pinned at exactly **−50.0** (its clip rail — the estimator is
  saturated and meaningless)
- `cos_sim(score, control)` ≈ **−0.1 to +0.08** throughout: the control is essentially orthogonal to
  the score. It is injecting noise, not steering.

**So the controller did not "activate". It went from collapsed (0.061%) to exploded (55,000%).
Neither is a working controller.** The v2 objective has no stable interior optimum.

### Why it diverged — and why gradient clipping could not save it

Two distinct mechanisms, in two phases. (Adversarial verification corrected my initial single-cause
account; the two-phase structure is the accurate one.)

**Phase 1 (epochs 1–5): no regulariser on ‖u‖ *at all*.** The `WarmupSchedule` sets
`path_kl_weight = 0` **and** `reinforce_weight = 0` for the first 5 epochs, while `detach_control_energy=True`
removes control energy from the gradient. The *only* gradient reaching the controller is the
trajectory-quality term — which is itself broken (B1 below). Control energy explodes 0.21 → 5301
(**25,000×**) before any Girsanov term is even switched on.

**Phase 2 (epoch 6+): the REINFORCE term is unbounded below.**
`loss = -mean(advantage · log_rn)` with a mean-zero advantage. Because the code's `log_rn` contains
`−½‖u‖²dt`, every sample with **below-average** reward (≈half of every batch, by construction of a
mean baseline) contributes `+A_i·½‖u_i‖²dt` with `A_i < 0` — **negative and growing quadratically
without bound in ‖u‖**. `total_loss` goes negative at exactly epoch 6, when `reinforce_weight` first
becomes non-zero. Additionally the advantage is **not** normalised by its standard deviation, and
rewards are **not** in `[0,1]` as the docstring claims — they reach ~1e6.

> **Gradient clipping bounds step *size*, not divergence.** On an objective unbounded below, clipped
> gradient descent still marches to infinity — just at a bounded rate per step. The observed *steady,
> monotonic* growth of the ratio (19.9 → 55,000 over 30 epochs) is precisely the signature of clipped
> descent on an unbounded objective. This is why `grad_clip=1.0` was active and irrelevant.

**Important:** fixing the Girsanov sign (A4) **does not by itself fix this**. With the corrected
`log_rn ~ +½‖u‖²dt/β`, the pathology *migrates* to positive-advantage samples (`-A·log_rn → -∞` for
`A > 0`). The REINFORCE surrogate needs advantage standardisation, reward normalisation, and a bounded
surrogate — not just a sign fix.

---

## Part C — Additional defects found

### C1. The trajectory quality loss has a silently broken gradient (`loss_components.py:139`)
```python
gen_sq = (gen_feats ** 2).sum(dim=1, keepdim=True)
with torch.no_grad():
    real_sq = (real_feats ** 2).sum(dim=1, keepdim=True)
    cross = gen_feats @ real_feats.T        # ← INSIDE no_grad: detached!
dist2 = gen_sq + real_sq.T - 2 * cross      # gradient flows ONLY through gen_sq
```
**Proof (executed):** the forward value is **bit-identical** to a correct `cdist` implementation, but
the gradient is exactly `2·gen_feats/(B·τ)` — the gradient of `‖φ(gen)‖²`, i.e. **pure shrinkage of
Inception features toward the origin**, with *zero* pull toward the nearest real sample. Cosine
similarity to the correct gradient: **0.74**. Confirmed bit-for-bit by three independent verifiers.

This is the load-bearing fix that Session 10 introduced to cure the collapse, and **it does not do
what it claims**. The nearest-neighbour search is 100% gradient-irrelevant (all columns share the
same gradient), and `clamp(min=0)` zeroes the gradient entirely on saturated entries.

### C2. `compute_v2_total_loss` adds control energy with **no weight coefficient** (`loss_components.py:270`)
`total = dsm + path_kl_weight*path_kl + quality_weight*traj_q + reinforce_weight*reinforce + ce`
— every other term has a weight; `ce` is bare (implicit 1.0), and the configured
`control_weight` is silently ignored. With `detach_control_energy=True` it contributes zero gradient
but pollutes the reported `total_loss`; with it `False`, it enters at a wrong, hardcoded weight.

### C3. Entropy production is computed on the **uncontrolled** trajectory
`scripts/compute_entropy_production.py:63` and `scripts/extended_baseline_analysis.py:121` (copy-paste)
The control is computed and tallied into the energy, but **never added to the drift**:
`drift = 0.5*beta_t*(-x - score)` — no `ctrl` term. So the reported "ITS entropy production" is the
uncontrolled path's. `data/results/entropy_production_profile.json` is void.

### C4. Two proven crashes in `scripts/extended_baseline_analysis.py`
- `:108` — `hasattr(ctrl_policy, "config")` is used to detect a ConvControlPolicy, but the MLP
  `ControlPolicy` **also** sets `self.config` (`neural_control.py:104`), so the 4-D branch is *always*
  taken. The documented command crashes.
- `:188` — `_load_or_train_classifier` loads `checkpoints/fmnist_classifier.pt` into `_TinyClassifier`,
  but `compute_mode_coverage.py` writes that **same path** with a **different architecture**. Collision → crash.

### C5. The Schrödinger-bridge / IPF solver cannot converge (`samplers/sb_solver.py`)
`compute_ipf_loss` is a pure `0.5/σ²·Σ‖u‖²dt` penalty whose unique minimiser is `u = 0`. There is
**no endpoint or marginal constraint anywhere**, so IPF as written cannot converge to a bridge.
`ipf_toy_convergence.json` measures nothing.

### C6. Latent bugs in the Girsanov path (found by verification, not yet triggered)
- `controlled_score_training.py:546` integrates `config.control_weight * control` into the drift, but
  `:544` uses **bare** `control_vec` — inconsistent whenever `control_weight ≠ 1`.
- `jarzynski_subsample > 1` sums only every N-th step **with no `1/N` rescaling** → silently
  under-counts path-KL. (Run 5A used subsample=1, so it was not triggered.)
- Train/eval **β mismatch**: training uses `beta_max=10.0`, `eval_controlled_multiseed.py:125` uses
  `beta_max=5.0`.

### C7. Documentation contains numbers that exist in no results file
`docs/controller_collapse_analysis.md:84` reports per-seed FIDs as *"(seed 42: 328.0, seed 1: 327.5,
seed 7: 327.0)"*. The actual values on disk are **seed 42: 327.11, seed 123: 326.12, seed 7: 329.19**
— different values, and a seed (`1`) that was never run. `docs/paper_requirements_roadmap.md:13`
asserts in the present indicative that *"The v2 objective redesign … fixes this"* — the run diverged.
`§8` lists *"Principled Girsanov SDE control formulation with path-KL regularization"* as
"Already Unconditionally Publishable"; the same document's §1.8 proves path-KL is the collapse
mechanism, and A4 above proves the Girsanov formula is wrong.

---

## Part D — The environment has regressed (blocks all training)

| | Recorded (`data/results/gpu_environment.json`) | **Actual, measured now** |
|---|---|---|
| torch | `2.5.1+cu121` | **`2.3.1+cpu`** |
| torchvision | — | **`0.18.1+cpu`** |
| `torch.version.cuda` | `12.1` | **`None`** |
| `cuda_available` | `true` | **`False`** |

The **hardware is present and healthy** — `nvidia-smi` reports a GTX 1650 Ti, 4096 MiB, driver 566.36,
CUDA 12.7, 0% utilisation. The *installed wheels* are CPU-only. Something reinstalled torch from the
default PyPI index.

**This is why run 5A took 176 hours for 39 epochs** — it trained on CPU. The one skipped test is
`test_phase5_hardening.py:1018: CUDA not available`.

**No Phase 2 training is realistic until this is fixed.** Extrapolating 5A's measured rate, a
100-epoch run costs ≈ **450 h ≈ 19 days** on CPU. The fix (not applied — requires confirmation, see
Phase 0):

```
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Note the 4 GB VRAM ceiling will constrain batch size regardless.

---

## Part E — Training run inventory

Target-epoch counts are from the launching script; "actual" is the last epoch in the JSONL log.

| Run | Dataset | Arch | Target ep | Actual ep | Evaluated? | Eval N | Seeds | Best FID | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `score_fmnist_v2` | FashionMNIST | ScoreUNet | 100 | **100** ✓ | yes (as baseline) | 5000 | 1 | DDPM 197.4 @NFE200 | **VOID** (A1/A2) |
| `controlled_v2_5a_seed42` | FashionMNIST | ConvControl | 100 | **39** ✗ | **NEVER** | — | — | — | **DIVERGED** |
| `controlled_v2_5a_pilot` | FashionMNIST | ConvControl | 20 | 20 | never | — | — | — | **DIVERGED** (ratio 367) |
| `controlled_v2_5b_joint` | FashionMNIST | ConvControl | 100 | **0** — never started | no | — | — | — | not run |
| `controlled_v2_5c_nopkL` | FashionMNIST | ConvControl | 100 | **0** — never started | no | — | — | — | not run |
| `controlled_config_d_seed42` | FashionMNIST | MLP | 60 | 60 ✓ | yes | **2048** ✗ | 3 | 327.47 ± 1.57 | **VOID** (A3: no control) |
| `controlled_conv_seed42` | FashionMNIST | ConvControl | 50 | 50 ✓ | yes | **2048** ✗ | 3 | 327.47 ± 1.57 | **VOID** — file is a duplicate |
| `ablation_{B,C,D,F,conv,no_EMA,no_freeze}` | FashionMNIST | mixed | 5 | 5 ✓ | yes | **2048** ✗ | 1 | all `326.0024719238281` | **VOID** (A3) |
| `sweep_fmnist/config_{A..F}` | FashionMNIST | MLP | 1 | 1 | yes | **256** ✗ | 1 | — | **VOID** (A3) |
| `ipf_fmnist_full` | FashionMNIST | SB policy | 5 | 3 ✗ | partial | — | — | — | **VOID** (C5) |
| `cifar10_*`, `score_cifar10_v2` | CIFAR-10 | — | — | **0** — dirs empty | random-weight only | **256** ✗ | 1 | 328.9 / 345.5 | **no CIFAR-10 results exist** |

### Paper-grade shortfall
The brief's bar is **N ≥ 5000 (FashionMNIST) / ≥ 10000 (CIFAR-10), ≥ 3 seeds**.

- **Exactly one file reaches N=5000**: `baseline_evals_fashionmnist.json` — and it has **1 seed**, not 3.
- **Everything else is at N=2048, 512, or 256.** This includes the entire
  `fmnist_eval_matrix_final.json` (24 entries at N=2048) that the project memory records as
  *"eval matrix done / paper-grade"*. **That record is wrong.**
- **CIFAR-10 has no real results at all** — `cifar10_baseline_eval.json` self-documents as
  *"Random-weight (--baseline-only) SDE evals. N=256"*, i.e. an untrained network.
- **Nothing anywhere meets the bar.** Even if the substrate were correct, zero runs are paper-grade.

### Checkpoints
- **Orphaned (trained, never evaluated):** `controlled_v2_5a_seed42` (ep 10/20/30 + last),
  `controlled_v2_5a_pilot` (ep 5/10/15/20 + last), `controlled_fmnist_test`, `score_fmnist_test`,
  `ipf_test`, `ipf_fmnist`, `score/` (50 files from a superseded run).
- **Empty dirs referenced by scripts/docs:** `controlled_v2_5b_joint`, `controlled_v2_5c_nopkL`,
  `cifar10_controlled_d_seed42`, `controlled_cifar10_test`, `score_cifar10_v2`, `sb_fmnist_test`,
  `suite/config_D_5ep`.
- **Missing but referenced:** `scripts/evaluate_model.py` — cited in
  `data/results/session_summary_session10.txt:96` as the Step-5D eval command. **It does not exist.**
  The real script is `scripts/eval_v2_multiseed.py`.
- All checkpoints referenced by results files are present on disk. No missing-checkpoint failures.

---

## Part F — What survives

Very little, but not nothing:

- **The infrastructure is sound.** Segmented/fault-tolerant training (exit-code-42 handoff, atomic
  autosave via `.tmp`+rename, `--resume-latest`, `--max-wall-hours`) is intact and functional. No long
  run need be lost to a crash.
- **Surface hygiene is genuinely clean**: 49/49 `torch.load` with `map_location`; all σ, √(1−ᾱ), β
  divisions clamped; `score_to_eps` correctly placed before both DDPM and DDIM updates; control
  reshape chain verified correct by execution; Jarzynski uses a stable log-sum-exp.
- **The DDPM/DDIM sampler is algebraically correct** (its problem is the A2 convention mismatch it
  inherits, not its own algebra).
- **The collapse *ratio* measurement** (`‖u‖/‖s‖` from training diagnostics) is a real measurement of
  a real training run — but of a run inside the A1 dynamics.
- **The thermodynamic diagnostic suite** (Crooks, Jarzynski, entropy production) is implemented and
  numerically careful; it is measuring the wrong trajectories (A1/C3), not measuring wrongly.

## Part G — Why ten sessions did not catch this

Worth recording, because it is the most transferable lesson here:

1. **Every defect is sign/convention-level, and every one produces plausible-looking output.** The
   path-KL is *positive* and *shaped like* `½‖u‖²dt` (two sign errors cancelling). The quality loss's
   forward value is *bit-exact*. The broken sampler produces *images*, and FID ≈ 325 looks like "a
   model that needs more training", not "a sampler integrating the wrong SDE".
2. **The test suite tests the wrong thing.** 77 tests, 76 passing — all shape/finiteness/importability
   assertions. Not one test would fail if the sampler sampled from `N(0,I)`. There is no test that
   the sampler recovers a known distribution, that control changes the output, or that path-KL equals
   the Girsanov KL.
3. **Identical numbers were never checked.** Six ablations reporting FID identical to 16 significant
   digits across *different architectures*, and a 10-point "learning curve" that is exactly constant,
   are impossible-by-inspection — and were reported as results across multiple sessions.
4. **Each session built on the previous session's conclusions rather than on its data.** The collapse
   narrative was inherited and elaborated (thermodynamic interpretation, phase-transition framing)
   without re-deriving the FID evidence it rested on.
