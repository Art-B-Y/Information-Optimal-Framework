# Negative Results and Lessons Learned

**Project:** ITS — Information-Theoretic Sampler
**Date:** 2026-04-18 · **Superseded in part:** 2026-07-15 audit

This document records experiments, configurations, and design decisions that did not work as
intended. Negative results are first-class scientific contributions; recording them here
prevents re-running failed approaches and motivates future work.

> ## ⚠️ Sections 1–N below predate the 2026-07-15 correctness audit
>
> Every experiment recorded below was run on a pipeline with seven proven defects (see
> `docs/current_state_diagnosis.md` and `data/results/phase2_preflight.txt`). **Their
> numbers are void**, and several of their *conclusions* are wrong — not merely
> imprecise. In particular, any entry that reasons from "ITS-SDE FID ≈ 333" or
> "controlled FID ≈ baseline FID" is reasoning from an artifact: control was silently
> disabled in every evaluation (`control_weight` defaulted to 0 and gated the control
> term), so the controlled and uncontrolled numbers were *literally the same
> computation*. Read them as a record of what was believed, not of what is true.
>
> **The genuine negative results of this project are §0 below.**

---

## 0. The real negative results (2026-07-15 audit)

These are the findings that survive, and they are about *method*, not about controlled
diffusion.

### 0.1 A green test suite certified a sampler that emitted noise

77 tests, 76 passing, for ten sessions — while the reverse-SDE drift had an inverted
sign, control was never applied, path-KL was wrong by a factor of β_t, and the DSM
objective was unweighted. **Not one test would have failed if the sampler returned
`N(0,I)`.** Every test asserted shapes, finiteness, or importability.

**Why it happened:** every defect was convention-level and produced *plausible* output.
Path-KL was positive and shaped like ½‖u‖²dt — because *two* sign errors cancelled. The
quality loss's forward value was bit-exact while its gradient pointed elsewhere (`cross`
was computed inside `torch.no_grad()`). A broken sampler still makes images, and FID≈325
reads as "needs more training", not "integrating the wrong SDE".

**The fix that matters:** `tests/test_scientific_correctness.py` asserts against
*analytically known* answers and asserts that error *decreases with compute*. Each test
was **falsified against the broken code** to prove it can fail.

**Sub-lesson, learned the hard way this session:** the first draft of the
REINFORCE-boundedness test *passed on the broken code* — it used random `log_probs`,
whose mean cancels, making it vacuous. A test that cannot fail on the bug it targets is
**worse than no test**: it certifies the bug as fixed. Always falsify.

### 0.2 Impossible numbers were reported as results for multiple sessions

- Six ablations (B, C, D, F, conv, no-EMA) reported FID identical to **16 significant
  digits** — *across different architectures*.
- A 10-point checkpoint "learning curve" (epochs 5→50) was **exactly constant** at
  335.1768493652344, while the `.pt` files were verifiably distinct.
- Two multiseed result files for *different models* were **byte-identical**.
- The SDE baseline reported FID 326.1 / 325.2 / 325.7 at NFE 50 / 100 / 200 — **flat**.
  A sampler that ignores compute is broken; this was visible on the surface for months.

**Lesson:** impossible-by-inspection results must be checked *as data*, not inherited as
conclusions. Each session built on the previous session's conclusions rather than its
numbers.

### 0.3 Gradient clipping cannot rescue an unbounded objective

Run 5A diverged to `total_loss = -1.59e14` with the control/score magnitude ratio
climbing 19.9 → 55,000 — **with `grad_clip=1.0` active**. Clipping bounds step *size*,
not divergence: on an objective unbounded below, clipped descent still marches to
infinity, just at a bounded rate. The *steady, monotonic* growth is that signature.
If a loss can be driven to −∞, no amount of clipping is a fix; bound the objective.

### 0.4 Fixing the headline bug was necessary but nowhere near sufficient

After Phase 1 fixed all four *fatal* defects, the corrected sampler still produced
**FID 206.49** (10 epochs, pilot). Three further defects had to be found before the
baseline was usable:

| config | change | FID @ 10 ep |
|---|---|---|
| A | Phase-1 end state (all 4 fatal defects fixed) | **206.49** |
| B | + σ_max 1.0 → 42 (valid VE prior) | **116.87** |
| C | + EDM preconditioning | **74.70** |

**Lesson:** "the big bug is fixed" is not the same as "the pipeline works". Only a
*gate* — an explicit, falsifiable criterion on the uncontrolled baseline — distinguishes
them. This is why Gate 2.2 exists and why it is a hard gate.

### 0.5 A prior session's optimisation diagnosis was itself wrong

The Speed session recorded `mixed_precision=False → no Tensor Cores` as a root cause of
an 8-day training failure. The GTX 1650 Ti is a TU117 die and **the GTX 16-series has no
tensor cores at all**; AMP is measured to be **3.7× slower** on it. The real cause of the
176-hour run was a CPU-only torch build. **Lesson:** measure the hardware; don't infer it.

---

## 1. FID on Training Split (Sessions 1–3)

**What we tried:** Evaluated DDPM, DDIM, and ITS-SDE on FashionMNIST using
`torchvision.datasets.FashionMNIST(train=True)` (60,000 images) as the real distribution.

**What we observed:** DDPM ≈ 361, DDIM ≈ 375, ITS-SDE ≈ 333 (N=256).

**Why it was wrong:** FID must be computed against the *test* split (`train=False`).
Using the training split over-estimates the reference distribution's quality, and the
comparisons between methods are misleading because all models were trained on these same images.

**Fix:** All reported results from Session 4 onward use `train=False` (10,000 images).

**Lesson:** Always explicitly pass `train=False` and log which split was used. Include it in
every result JSON as `"eval_split": "test"`.

---

## 2. N=256 FID Estimates Are Unreliable

**What we tried:** Used N=256 generated samples for FID across all sweep configs.

**What we observed:** Six different hyperparameter configurations (ctrl_w ∈ {0.001, 0.01, 0.1},
pkl_w ∈ {0.05, 0.1, 0.5}, q_w ∈ {0.0, 0.01, 0.05}) all produced FID within 430–450.
This appeared to show zero hyperparameter sensitivity, which would be a null result.

**Why it failed:** At N=256, FID variance is approximately 30–80 points (from Monte Carlo
estimation of the Fréchet distance between two Gaussian fits each from 256 samples).
The spread between configurations was smaller than this noise floor, making discrimination
impossible.

**Fix:** Use N≥2048 for all reported results (Session 4+). N=5000 is standard in the
diffusion model literature; N=2048 is acceptable for ablation studies.

**Lesson:** Always state N alongside every FID number. Never compare FID values across
papers or configs unless N matches or a correction is applied.

---

## 3. Zero-Initialized Control Policy After Short Training

**What we tried:** 1-epoch / 2000-sample controlled training with control_weight=0.01.

**What we observed:** Control energy ≈ 0.04 (near zero). Score model weight ratio ≈ 1.0
vs. random initialization (indicating the score model barely moved). FID indistinguishable
from DDPM without control.

**Why it happened:** (a) 2000 samples / 64 batch size = 31 gradient steps is insufficient
for the score model to converge; (b) the control policy is zero-initialized, so it starts
as a pure DDPM/SDE sampler; (c) control_weight=0.01 may be too small to produce detectable
signal in ≤100 steps.

**Fix:** Full-dataset training for ≥5 epochs is the minimum for meaningful control learning.
On Google Cloud GPU, 10 epochs on 60k FashionMNIST is feasible.

**Lesson:** Always report training data volume (N_train × epochs) alongside evaluation metrics.

---

## 4. Hydra Struct Mode Rejects Unknown Config Keys

**What we tried:** Override `dataset_subset_size` and `early_stop_patience` via Hydra CLI
(`training.dataset_subset_size=10000`) without declaring them in the YAML.

**What we observed:**
```
omegaconf.errors.ConfigAttributeError: Key 'dataset_subset_size' is not in struct
```

**Why it happened:** Hydra's struct enforcement prevents adding keys not present in the
base config, preventing accidental typos from silently being ignored. This is correct
behavior but requires all new fields to be added to the YAML before they can be overridden.

**Fix:** Added `dataset_subset_size: 0`, `early_stop_patience: 0`, `early_stop_min_delta: 0.01`
to both `configs/experiment/controlled_mnist.yaml` and `configs/experiment/controlled_cifar.yaml`.

**Lesson:** New training config fields must be added to both the dataclass
(`ControlledScoreTrainingConfig`) and the YAML simultaneously.

---

## 5. DDPM Numerical Instability at t=0

**What we tried:** Used `torch.sqrt(1 - alpha_bar_t)` directly in `ddpm.py` without clamping.

**What we observed:** Potential NaN from `sqrt(negative_value)` when `alpha_bar_t ≈ 1.0`
at the first timestep due to floating-point rounding.

**Fix:**
```python
one_minus_ab = (1.0 - alpha_bar_t).clamp(min=1e-5)
sigma_t = torch.sqrt(one_minus_ab)
```
Applied consistently to all `sqrt(1 - alpha_bar)` expressions in the DDPM and DDIM samplers.

**Lesson:** Always clamp before `sqrt` when the argument is a difference that should be
non-negative but may be slightly negative due to floating-point.

---

## 6. Schrodinger Bridge / IPF: Phase Alternation Not Verified

**What we tried:** Ran the IPF driver for 2 iterations on 500-sample FashionMNIST subset.

**What we observed:** The script executed without error, but the background process
was lost due to a session crash before verification of: (a) phase alternation
(forward/backward phases logged distinctly), (b) non-zero ipf_loss, (c) checkpoint
containing both policy dicts.

**Status:** *Unverified.* The IPF/SB implementation in `src/its/experiments/sb_fmnist.py`
has not been end-to-end tested with multi-phase alternation confirmed in logs.

**Recommended follow-up:** Run `python scripts/train_schrodinger_bridge.py --iterations 2 --subset 500`
and explicitly verify the JSONL log shows `{"phase": "forward"}` and `{"phase": "backward"}`
entries, and that `ipf_loss` is non-NaN and non-zero.

---

## 7. Feature-Matching Quality Loss: Uncertain Effect

**What we tried:** Added `quality_weight=0.01` to enable Inception-v3 feature-matching loss
as an auxiliary objective during control policy training.

**What we observed:** No FID improvement verifiable at N=256 noise levels (see §2). At
N=2048 with longer training, the effect remains untested.

**Theoretical concern:** Feature-matching in pixel space via Inception-v3 features
introduces a mode-covering pressure that may conflict with the KL minimization objective.
Specifically, Inception-v3 was pretrained on ImageNet; its features may not be diagnostic
for FashionMNIST quality.

**Recommended follow-up:** Ablation comparing `quality_weight=0.0` vs `quality_weight=0.01`
at N=2048 with fixed random seed and ≥5 epochs. Without this ablation, quality_weight
should not be claimed to help in the paper.

---

## 8. Large Score Gradient Norms at Initialization

**What we observed:** Score model gradient norms of ~3.5×10^6 at step 0, decaying to
~7×10^5 by step 4. These triggered `WARNING High score grad norm` repeatedly.

**Why it happens:** Random Kaiming initialization of a UNet with skip connections can
produce large initial gradients, especially with unnormalized input. This is expected
and the model converges.

**Not a bug:** The norms decrease monotonically and training does not diverge. However,
gradient clipping (`torch.nn.utils.clip_grad_norm_`) with `max_norm=1.0` would suppress
these warnings and is standard practice for diffusion model training.

**Recommended follow-up:** Add `grad_clip_norm: 1.0` option to training config.

---

## Summary Table

| # | Issue | Status | Impact |
|---|-------|--------|--------|
| 1 | FID on train split | Fixed (Session 4) | All Sessions 1–3 FID invalid |
| 2 | N=256 FID variance | Fixed (Session 4) | Use N≥2048 |
| 3 | Zero-init control / short training | Partially fixed | Need ≥5 ep full data |
| 4 | Hydra struct violation | Fixed | YAML fields added |
| 5 | DDPM sqrt NaN | Fixed | Numerical stability |
| 6 | IPF phase alternation | Unverified | Need re-run |
| 7 | Feature-matching loss effect | Unknown | Ablation needed |
| 8 | Large initial grad norms | Benign / Known | Optional clip |
