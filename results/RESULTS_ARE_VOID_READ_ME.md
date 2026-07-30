# ⛔ EVERY PRE-2026-07-15 RESULT IN THIS DIRECTORY IS VOID

> ## ✅ SUPERSEDED BY (2026-07-16, Phase 2) — use these instead
>
> | void artifact | **authoritative replacement** |
> |---|---|
> | `baseline_evals_fashionmnist.json` (SDE FID 326.1/325.2/325.7, flat; N=5000 **1 seed**) | **`gate_2_2_report.json`** — FID **111.6/53.0/31.3/26.0** at NFE 50/100/200/500, *improving*; **19.71 ± 0.32** at N=5000 × **3 seeds** |
> | `fmnist_eval_matrix_final.json` (24 rows, N=2048 — recorded as "paper-grade"; it is not) | `gate_2_2_report.json` (N=5000, 3 seeds — the project's **first** paper-grade eval) |
> | `controlled_config_d_multiseed.json`, `controlled_conv_multiseed.json` (FID 327.47; byte-identical to each other) | *pending* — controller not yet retrained on the corrected substrate |
> | `ablation_study.json` (all six FID `326.0024719238281`) | *pending* — needs re-running with control actually applied |
> | `conv_checkpoint_fid_trajectory.json` (all ten epochs `335.1768493652344`) | `gate_2_2_report.json` §2 — a **real** FID-vs-epoch curve: 98.5 → 53.0 |
> | `entropy_production_profile.json` (control never in the drift) | *pending* — Step 4 |
> | `data/paper_figures/*` (all pre-audit) | `baseline_fid_vs_epoch.{png,pdf}`, `baseline_fid_vs_nfe.{png,pdf}` |
> | `data/results/final_samples*.png`, `samples_sde_*` | **`baseline_final_samples.png`** — recognisable clothing, not noise |
> | checkpoints `score_fmnist_v2/`, `score/`, `controlled_*`, `ablation_*`, `sweep_*` | **`checkpoints/fmnist_score_corrected_baseline/`** (canonical) |
>
> **Full report:** [`docs/gate_2_2_report.md`](../../docs/gate_2_2_report.md).
> Anything marked *pending* has no valid replacement yet — do not substitute the void number.
>
> **Also note:** even the corrected FIDs are **not comparable** to the pre-audit ones beyond
> the sampler fix — FID itself was computed on *inverted* images (`normalize=True` fed
> uint8), so the old numbers live in a different pixel space entirely.

---

## Original notice (2026-07-15 audit)

**Do not use any number, figure, or table in `data/results/` or `data/paper_figures/` for the paper,
for a comparison, or as a baseline.** They were produced by code with four independent, individually
fatal defects. Each defect below has an executed proof in `docs/current_state_diagnosis.md`.

Nothing has been deleted or moved. This notice is additive so the record of what was found is
preserved. **Quarantining these files into `deprecated_broken_sampler/` is Phase 1E and has NOT been
done** — it awaits confirmation, because moving a project's entire result history is the user's call,
not mine.

---

## Why they are void

| # | Defect | Effect on results |
|---|--------|-------------------|
| **A1** | The score-SDE sampler integrates the wrong dynamics (sign inverted on both drift terms, factor 2 on the score, positive step while `t` descends). With an exact analytic score it converges to **~84× the true variance** and **never improves with NFE**. | Every FID from the SDE sampler is meaningless. Matches the observed flat SDE FID **326.1 / 325.2 / 325.7** at NFE 50/100/200. |
| **A2** | The score model is trained with a **VE** kernel (`x + σ·ε`) but sampled with **VP** dynamics. Incompatible diffusion families. | Even the DDPM baseline (FID 197) is depressed by the convention mismatch; a healthy FashionMNIST model reaches FID < 20. |
| **A3** | `control_weight` defaults to `0.0` and gates the control branch; **no eval call-site ever passed it**. | **Every "controlled" result is the uncontrolled sampler.** The control policy was loaded, passed in, and never applied. |
| **A4** | The Girsanov/path-KL formula is wrong by a factor of **β_t**, which is time-varying (β sweeps 0.1→10) and so cannot be absorbed into a weight. | Every path-KL number ever reported is wrong. |

## The tells that were visible on the surface all along

- `ablation_study.json` — B, C, D, F, conv, no-EMA all report `fid = 326.0024719238281` **identical to
  16 significant digits, across different architectures**. Impossible unless control never applied.
- `conv_checkpoint_fid_trajectory.json` — all 10 checkpoints (epochs 5→50) report
  `fid = 335.1768493652344`. A learning curve that is *exactly constant*. The `.pt` files are distinct.
- `controlled_config_d_multiseed.json` and `controlled_conv_multiseed.json` — **byte-identical**
  `runs` and `ci_95` blocks, while claiming to be different models (MLP config D @60ep vs
  ConvControlPolicy @50ep). One is a copy of the other.
- `entropy_production_profile.json` — the control was tallied into the energy but **never added to the
  drift** (`compute_entropy_production.py:63`), so this profiles the *uncontrolled* path.
- `ipf_toy_convergence.json` — `sb_solver.compute_ipf_loss` is a pure `‖u‖²` penalty with **no marginal
  constraint**; its unique minimiser is `u=0`, so IPF cannot converge to a bridge. It measures nothing.
- `cifar10_baseline_eval.json` — self-documents as *"Random-weight (--baseline-only) SDE evals, N=256"*.
  **No CIFAR-10 result has ever existed.**

## Also: nothing here was ever paper-grade

The bar is N ≥ 5000 (FashionMNIST) / N ≥ 10000 (CIFAR-10), ≥ 3 seeds.

- **Exactly one file reaches N=5000** — `baseline_evals_fashionmnist.json` — and it has **1 seed**.
- Everything else is **N=2048, 512, or 256**, including all 24 entries of
  `fmnist_eval_matrix_final.json`, which the project memory records as *"eval matrix done,
  paper-grade"*. **That record is wrong.**

## What this means for the paper's planned contribution

The "controller collapse" finding — the project's headline result across Sessions 9–10 — rested on
two legs, and **both fail**:

1. *"Controlled FID equals baseline FID ⇒ the controller collapsed."* **Invalid.** Those FIDs are
   identical because they are literally the same computation (A3).
2. *"‖u‖/‖s‖ = 0.061% at epoch 50."* A **real** measurement from training diagnostics (which do apply
   control) — but measured inside dynamics that never generated data (A1).

The collapse may well reproduce on a correct substrate; the path-KL-minimised-at-`u=0` argument is
sound in principle. But it must be **re-established**, not assumed. See `docs/roadmap_current.md`.

## What is NOT void

- `data/logs/controlled_v2_5a_seed42.jsonl` — a genuine record of a genuine divergence. Retain it;
  it is the evidence for the unbounded-objective analysis (Part B of the diagnosis).
- `data/results/gradient_flow_diagnosis.json`, `collapse_trajectory_metrics.json`,
  `loss_term_magnitudes.json` — real measurements of real training runs, but of runs conducted inside
  the A1 dynamics. Interpret with that caveat; do not treat as evidence about controlled diffusion.
- `_pre_phase1_backup_20260715/` — the pre-audit `src/`, `scripts/`, `tests/` trees. This project is
  **not a git repository**, so this is the only undo path for the Phase 1 edits.

---
*Generated by the 2026-07-15 audit session. See `docs/current_state_diagnosis.md` for proofs,
`docs/roadmap_current.md` for the recovery plan, `data/results/audit_phase1.txt` for the fix log.*
