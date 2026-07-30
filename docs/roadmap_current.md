# ITS — Roadmap to a Publishable Paper

**Written:** 2026-07-15 · **Regenerated:** 2026-07-16 after Phase 2
**Status:** ✅ Phase 0 · ✅ Phase 1 · ✅ **Phase 2 — GATE 2.2 PASSED (5/5)** · Phase 3 next
**Grounded in:** `docs/current_state_diagnosis.md`, `data/results/audit_phase1.txt`,
`data/results/phase2_preflight.txt`, `docs/gate_2_2_report.md`

> ## 🎉 The project has a working generative model for the first time
>
> **Corrected baseline: FID = 19.71 ± 0.32** (N=5000, 3 seeds, NFE=500) — the first
> paper-grade evaluation in the project's history.
>
> | | pre-audit (void) | corrected |
> |---|---|---|
> | FID @ NFE 50/100/200 | 326.1 / 325.2 / 325.7 — **flat** | 111.6 / 53.0 / **31.3** — improving |
> | paper-grade FID | never done | **19.71 ± 0.32** |
> | samples | noise | recognisable clothing |
>
> **Eight defects fixed in total** (4 fatal in Phase 1 + 4 more in Phase 2). Phase 1 alone
> was **not sufficient**: its end state still gave FID 206.

> **This roadmap replaces the previous plan.** The prior roadmap assumed the baseline evaluation
> matrix was paper-grade and that the only open question was whether the v2 objective would activate
> the controller. Both assumptions are false. **Every experimental result in the project is void**
> (4 independent fatal defects), **nothing has ever been evaluated at paper-grade sample count**, and
> **the v2 run diverged rather than being pending.** The project is further from a paper than the
> record suggested, and the honest plan is longer.

---

## Phase 0 — Unblock the environment ✅ **COMPLETE**

CUDA restored: `torch 2.5.1+cu121`, GTX 1650 Ti (4.3 GB, CC 7.5), 1.72 TFLOP/s FP32 warm.
The previously-skipped CUDA test now runs and passes (89 passed, 0 skipped).

**Hardware finding that changed the plan:** **AMP is 3.7× SLOWER** on this GPU (bs=256:
1297 img/s FP32 vs 346 img/s AMP). The GTX 1650 Ti is a TU117 die and **the GTX 16-series
has no tensor cores**. Phase 2 therefore trains in FP32, deviating from the brief's AMP
instruction — that instruction is premised on hardware behaviour that does not hold here.
*Corollary:* the Speed session's recorded root cause ("mixed_precision=False → no Tensor
Cores") was itself a misdiagnosis; the 176-hour run was caused by the CPU-only torch build.

---

## Phase 1 — Technical error elimination ✅ **COMPLETE** `[NO RESULTS PRODUCED]`

Full fix log: **`data/results/audit_phase1.txt`**. Every fix verified by execution.

**The brief's original checklist was audited and found already clean** — imports, all 49 `torch.load`
`map_location` calls, device consistency, shape mismatches, numerical guards (σ, √(1−ᾱ), β all
clamped), `score_to_eps` placement, config completeness, the segmented infrastructure. Prior sessions
did that work properly. **Every real defect was semantic**, and every one produced plausible-looking
output — which is exactly why ten sessions missed them.

### Fixed
| # | Defect | Fix | Verification |
|---|---|---|---|
| **A1/A2** | Sampler integrated the wrong SDE (sign inverted, factor 2 on score, VE-trained/VP-sampled) | Correct **VE reverse-diffusion** predictor in both the eval sampler and `simulate_path`; `parameterisation` flag (`"ve"` default, `"legacy_vp"` for archive reproduction only); VE prior init `N(0,σ_max²)` | Analytic score: **mean 2.97 / std 0.5046 → target 3.0 / 0.5, and improves with NFE**. Legacy: mean −74, worsening. |
| **A3** | `control_weight=0.0` default silently disabled control in every eval | `sample()` now **raises** if a policy is passed with `control_weight≤0`; eval uses `dataclasses.replace` so no field can be dropped; `control_weight=1.0` explicit; β aligned 5.0→10.0 | Test asserts control changes the sample; test asserts the raise |
| **A4** | Girsanov formula wrong by factor β_t (4 copies) | New **`src/its/sde/girsanov.py`** — one implementation, stated in realised drift-shift and noise-variance terms (valid for VP/VE/any Euler). All 4 sites use it. **Sign flipped jointly** with the formula. | Matches exact Gaussian log-densities to **3.3e-16**; MC-verified to 4.4e-15 |
| **B2** | REINFORCE unbounded below → the 5A divergence | Standardised advantage (scale-free), clamp advantage ±5 and log-prob ±50; clips apply to the score-function *coefficient*, so the estimator stays an unbiased policy gradient in the unclipped region | Test falsified: pre-audit reaches **−9.94e7**; fixed version bounded |
| **C1** | Quality gradient silently broken (`cross` inside `no_grad`) | Moved `cross` out of the `no_grad` block | Gradient now matches finite differences; was exactly `2·gen_feats/(B·τ)` (pure feature-norm shrinkage) |
| **C2** | Control energy added with implicit weight 1.0 | Explicit `control_energy_weight` | Test asserts no leak at weight 0 |
| **C3** | Entropy production walked the **uncontrolled** trajectory | Control term integrated into the drift | — |
| **C6** | Girsanov got bare `control_vec` while the state got `control_weight*control`; `jarzynski_subsample` un-rescaled; Jarzynski pinned at its ±50 rail | Exponent receives the exact integrated tensor; subsample rescaled; `jarzynski_saturated` flag emitted | Smoke test: Jarzynski now returns **−1.008**, not the −50 rail |

### 1D — The test suite can now fail on broken science ✅ *(the highest-leverage item)*
`tests/test_scientific_correctness.py` — **12 tests, each falsified against the pre-audit code.**
Three confirmed kills: the sampler test (pre-audit mean **−1080** vs target +3.0), the Girsanov test
(**0.6162** vs exact 0.7190), the REINFORCE-boundedness test (**−9.94e7**).

> **A finding worth carrying forward:** the first draft of the REINFORCE test **passed on the broken
> code** — it used random `log_probs`, whose mean cancels, so it was vacuous. Falsification caught it
> and it was rewritten to model the real geometry. *A test that cannot fail on the bug it targets is
> worse than no test: it certifies the bug as fixed.* That is the mechanism by which this project
> accumulated 77 green tests over a sampler that emitted noise.

### Still open (documented, not hidden — see audit Part 3)
- **C4** — two proven crashes in `extended_baseline_analysis.py` (`:108` always-true `hasattr`
  dispatch, `:188` classifier checkpoint collision). Deferred: produces nothing needed before Phase 2.
- **C5** — `sb_solver.compute_ipf_loss` has no marginal constraint, so IPF cannot converge to a
  bridge. Needs a design decision: implement the constraint, or declare IPF out of scope.
- **C7** — `docs/controller_collapse_analysis.md:84` reports per-seed FIDs that exist in no results
  file (and a seed never run); `paper_requirements_roadmap.md` asserts the v2 redesign "fixes this"
  (it diverged). These describe a superseded understanding; rewriting belongs with the paper.
- **1E** — quarantining the void results into `deprecated_broken_sampler/`. **Not done:** moving a
  project's entire result history is the user's call. An additive, non-destructive notice was written
  instead: **`data/results/RESULTS_ARE_VOID_READ_ME.md`**.

**Backup:** this project is **not a git repository**. The pre-audit `src/`, `scripts/`, `tests/` trees
are at `data/results/_pre_phase1_backup_20260715/` — the only undo path.

- **Cost:** 0 GPU-h. **Dependency:** none.

---

## Phase 2 — Retrain the baseline + Gate 2.2 ✅ **COMPLETE**

### 2.1 Corrected baseline ✅
`fmnist_score_corrected_baseline`, 100 epochs, **2.1 h wall clock** (vs 176 h for 5A on CPU).
VE kernel ↔ VE sampler, σ∈[0.01,42], σ²-weighted DSM, EDM preconditioning, FP32, bs=256.
The segmented launcher handed off cleanly mid-run (segment 1 exit-42 at 1.51 h / epoch 73;
segment 2 auto-resumed) — fault tolerance exercised live, not assumed.

### 2.2 GATE ✅ **PASS 5/5** → `docs/gate_2_2_report.md`
| criterion | result |
|---|---|
| 1 stable training | ✅ loss 0.166→0.135, grad max 2.31, val tracks train (ratio 1.02) |
| 2 FID ↓ with training | ✅ 98.5 → **53.0** (epochs 10→100) |
| 3 FID ↓ with NFE | ✅ **111.6 → 53.0 → 31.3 → 26.0** (NFE 50→500) |
| 4 competitive FID | ✅ **19.71 ± 0.32** (N=5000, 3 seeds) — target <30 |
| 5 visual | ✅ recognisable trousers, pullovers, dresses, sneakers, boots, bags |

**Four further defects were found and fixed here** — Phase 1's four fatal fixes were
necessary but nowhere near sufficient (its end state still gave FID 206):

| defect | evidence |
|---|---|
| σ_max=1.0 was ~42× too small for a VE prior (true value 41.88) | pilot **206.49 → 116.87** |
| DSM objective unweighted (loss ∝ 1/σ², 1.7e7 ratio) | gradient collapsed onto small σ |
| preconditioning `h = x/σ` spans **70×** input scale | pilot **116.87 → 74.70** |
| EMA had no bias correction (80% random init at epoch 10) | epoch 10: RAW **99.75** vs EMA **373.07** |

### 2.3 Controller pre-flight ✅ (bounded-objective probe)
`data/results/controller_bounded_probe.json` — 30 steps on the real config:

| | run 5A | now |
|---|---|---|
| total_loss | → −1.59e14 | bounded **[−0.15, +0.90]** |
| ‖u‖/‖s‖ | 19.9 → 55,000 | 0.0010 → **0.0057** |
| path_kl ≥ 0 | violated | **always** ✓ |

Phase 1's REINFORCE and Girsanov fixes hold on the real configuration.

**Remaining in Phase 2 — NOT DONE (see "next action"):** the full controller run
(`scripts/train_controlled_corrected.py`, gate-guarded, ready to launch).

---

## Phase 3 — Evaluations at paper grade `[DEPENDS: 2]`

Every completed run evaluated at **N=5000 (FashionMNIST) / N=10000 (CIFAR-10), 3 seeds (42/123/7)**,
95% CI. **This has never once been done in this project** — the record's "paper-grade matrix" is N=2048.

Matrix: {DDPM, DDIM, SDE-uncontrolled, ITS-v1, ITS-v3, ablation} × NFE {25, 50, 100, 200} × 3 seeds.

- **Guard:** assert no two rows are bit-identical (the A3 signature).
- **Deliverable:** `fmnist_eval_matrix_v2.json`. **Cost:** ~15–20 GPU-h.
- CIFAR-10 is **new work, not a re-run** — no CIFAR-10 result has ever existed. Descope it unless
  FashionMNIST lands cleanly (+~40 GPU-h and a bigger model than 4 GB comfortably holds).

---

## Phase 4 — Scientific analyses `[DEPENDS: 3]`

Only meaningful once 2.2 passes. Pareto frontier (quality/energy/information/compute); Crooks
verification; Jarzynski validity (**check saturation — it pinned at the ±50 clip rail in 5A**);
entropy-production profiles (**with control actually in the drift**); control-drift analysis; mode
coverage; memorization check. Each at statistically meaningful N.
**Cost:** ~10–15 GPU-h.

---

## Phase 5 — Paper artifacts `[DEPENDS: 4]`

LaTeX results table, publication figures, reproducibility manifest, analytics report — all
**regenerated from scratch**. Existing figures in `data/paper_figures/` are void.
Add a manifest field recording the substrate-correctness gate (2.2) result.
**Cost:** ~1 GPU-h.

---

## Phase 6 — Paper framing and venue `[DEPENDS: 2–4]`

**This decision cannot be made now, and the previous roadmap's decision tree is unusable** — it
branched on results that do not exist. Framing follows the data:

| If Phase 2–4 shows | Contribution | Venue |
|---|---|---|
| Controller gives a real Pareto improvement over a **healthy** baseline (FID < 30) | Positive: controlled sampling improves the quality/compute/information trade-off | NeurIPS/ICML/ICLR main track |
| Controller matches baseline quality at lower NFE or lower path-KL | Efficiency/information contribution | Workshop → main track |
| Controller genuinely collapses **on a correct substrate** | *Now* a real negative result: collapse is intrinsic to the objective, with thermodynamic analysis | TMLR / specialised venue |
| Controller cannot be stabilised at all | Honest negative result + the failure-analysis methodology | TMLR |

**The current honest position:** the collapse claim is **not yet evidence of anything**, because the
FID half of its argument was control-never-applied and the ratio half was measured inside a sampler
that integrates the wrong SDE. It must be **re-established on a correct substrate** before it can be
a contribution. It may well reproduce — the path-KL-minimised-at-`u=0` argument is sound in
principle — but that must be shown, not assumed.

**One contribution survives independently of every re-run:** the failure analysis itself. Four
convention-level defects, each producing plausible output, surviving ten sessions and a 77-test suite,
because the tests checked shapes and not science. That is a real lesson about validating scientific
ML software and is publishable as a case study — but it is a lesson about *method*, not about
controlled diffusion, and it should not be mistaken for the paper the project set out to write.

---

## Critical path & totals

```
Phase 0 (env) ─┐
               ├─> Phase 2 ─> [GATE 2.2] ─> Phase 3 ─> Phase 4 ─> Phase 5 ─> Phase 6
Phase 1 (code)─┘
```

| Phase | GPU-h | Depends |
|---|---|---|
| 0 env | ~0 | — |
| 1 code | ~0 | — |
| 2 training | 80–110 | 0, 1 |
| 3 evals | 15–20 | 2 |
| 4 analyses | 10–15 | 3 |
| 5 artifacts | ~1 | 4 |
| **Total** | **~105–145 GPU-h** | |

≈ **2–4 weeks wall-clock** on the local 1650 Ti; ≈ **3–5 days** on rented A100s.

## The single most important next action

**Launch the controller run.** Gate 2.2 has passed, the objective is proven bounded, and the
script is gate-guarded and ready:

```bash
python scripts/launch_segmented_training.py \
    --script scripts/train_controlled_corrected.py \
    --target-epochs 50 --checkpoint-dir checkpoints/fmnist_controlled_corrected \
    --max-segment-hours 3
```

**Budget it honestly — this is the expensive one.** Each step back-propagates through a
20-step differentiable SDE rollout *plus* Inception, at bs=32 → ~1875 steps/epoch. Expect
**~20–40 min/epoch**, i.e. **~20–30 GPU-h for 50 epochs (1–2 days wall-clock)** on the
1650 Ti. This is the single largest remaining time risk; renting an A100 (~$2/h) would
compress it to a few hours. **Use the segmented launcher** — a run this long must not be
lost.

**Watch `ctrl_score_mag_ratio` every epoch.** It is the primary indicator:
- *exploding* (5A's failure: 19.9 → 55,000) — should now be impossible; the objective is
  bounded and the probe held at 0.006 over 30 steps.
- *collapsing to ~0* — the v1 failure mode. **Now genuinely interpretable for the first
  time**, because the sampler works and the control is actually applied. If the controller
  collapses on a *correct* substrate, that is a real negative result and a real paper.

**Then Step 4** (thermodynamics) — only meaningful now: control energy, entropy production
(controlled vs uncontrolled), corrected path-KL, Jarzynski with its validity check
(`std(work)/kT`; it pinned at the ±50 clip rail in 5A), Crooks verification, and the fair
**equal-NFE** controlled-vs-baseline comparison at N=5000 / 3 seeds.

> **Compare at equal NFE.** The baseline is 19.71 at NFE=500 but **52.96 at NFE=100**. A
> controlled-vs-baseline comparison at mismatched NFE would be meaningless.
