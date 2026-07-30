# Gate 2.2 — Baseline Validation Report

**Date:** 2026-07-16 · **Run:** `fmnist_score_corrected_baseline` · **Verdict: ✅ PASS (5/5)**

> **Headline: FID = 19.71 ± 0.32** (N=5000, 3 seeds, NFE=500) on FashionMNIST.
>
> **This is the first time in the project's history that the pipeline produces a working
> generative model.** The pre-audit sampler produced FID ≈ 325, flat across every NFE.

Gate 2.2 is the decisive checkpoint of the project. No ITS number means anything until the
**uncontrolled** baseline is proven to work. Its absence is what allowed ten sessions to
build a thermodynamic interpretation on top of a sampler that emitted noise.

The controller (`scripts/train_controlled_corrected.py`) **refuses to run in code** unless
this report records a pass.

---

## The comparison that matters

| | pre-audit (void) | corrected |
|---|---|---|
| FID @ NFE=50 | 326.09 | **111.64** |
| FID @ NFE=100 | 325.20 | **52.96** |
| FID @ NFE=200 | 325.69 | **31.31** |
| FID @ NFE=500 | — | **25.98** |
| **paper-grade (N=5000, 3 seeds)** | never done | **19.71 ± 0.32** |
| improves with compute? | **no — flat** | **yes, monotonically** |
| samples | noise | recognisable clothing |

A sampler whose FID does not move when you give it 4× the compute is broken. That was
visible on the surface of `baseline_evals_fashionmnist.json` for months.

---

## Criterion 1 — Stable score-model training ✅ PASS

| metric | value |
|---|---|
| train DSM loss (first quarter → last quarter) | 0.1661 → **0.1352** |
| all losses finite | yes — no NaN, no divergence |
| max gradient norm over the whole run | **2.31** (bounded; clip=1.0) |
| val DSM loss (epoch 1 → 100) | 0.2220 → **0.1367** |
| overfit ratio (val/train), last | **1.02** — val tracks train |
| overfit ratio, max | 1.06 — never approached the 1.2 warning threshold |

Training was stable throughout. Loss fell monotonically in trend and the validation loss
tracked it without diverging, so there is no meaningful overfitting.

## Criterion 2 — FID decreases during training ✅ PASS

| epoch | 10 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|
| FID | 98.46 | 102.20 | 66.39 | 66.39 | **52.96** |

(N=2048, seed 42, NFE=100.) 98.46 → 52.96 — FID improves with training.

**Two honesty notes:**
1. The curve is **not monotone** (epoch 30 is slightly worse than epoch 10). With a single
   seed at N=2048 the FID noise is a few points, so this is within noise of a plateau; it
   is not evidence of instability, and the endpoints are unambiguous.
2. **Epochs 50 and 70 print as identical (66.39/66.39).** That is exactly the
   impossible-number signature the audit flagged (`conv_checkpoint_fid_trajectory.json`
   had *all ten* epochs bit-identical), so it was checked rather than assumed. The
   checkpoints are genuinely distinct (different md5; L2 distance 3.14 between weight
   vectors) and the full-precision FIDs differ: **66.39476776** vs **66.38622283**. It is
   a `%.2f` display coincidence on a plateau, not a bug.

**FID was still falling at epoch 100** (66.4 → 53.0), so the model is **undertrained**, not
converged. More epochs would improve it further. See "Limitations".

Figure: `data/paper_figures/baseline_fid_vs_epoch.{png,pdf}`

## Criterion 3 — FID improves with larger NFE ✅ PASS *(the most important criterion)*

| NFE | 50 | 100 | 200 | 500 |
|---|---|---|---|---|
| FID | 111.64 | 52.96 | 31.31 | **25.98** |

(best checkpoint = epoch 100, N=2048, seed 42.)

**A 4.3× monotone improvement from NFE 50 → 500.** This is the criterion whose failure was
the clearest symptom of the broken sampler: the void results showed FID **flat at ~326**
across NFE 50/100/200, which is impossible for a working sampler — more sampling steps must
buy accuracy. The corrected VE sampler now behaves correctly.

Figure: `data/paper_figures/baseline_fid_vs_nfe.{png,pdf}`

## Criterion 4 — Competitive FID at paper grade ✅ PASS

| seed | 42 | 123 | 7 |
|---|---|---|---|
| FID | 19.81 | 19.36 | 19.97 |

**FID = 19.71 ± 0.32** — N=**5000**, **3 seeds**, NFE=500, epoch-100 checkpoint.
Target: < 30. **Paper-grade: yes** (≥5000 samples, ≥3 seeds — the first paper-grade
evaluation this project has ever produced).

The seed spread is tight (±0.32), so the number is stable. Low-double-digit FID is
competitive for FashionMNIST with a 1.6M-parameter UNet.

## Criterion 5 — Samples are visually correct ✅ PASS

Grid: `data/results/baseline_final_samples.png` (128 samples, epoch 100, NFE=500).

**Inspected.** The grid shows unambiguously recognisable FashionMNIST items: trousers,
pullovers and long-sleeve tops, dresses, coats, sneakers, sandals, ankle boots, and
handbags with visible handles. Diverse across classes with no mode collapse. Not noise,
not uniform blobs, not static.

Sample statistics closely match the real data:

| | generated | real data |
|---|---|---|
| mean | −0.330 | −0.431 |
| std | 0.737 | 0.702 |

---

## What made the difference

Phase 1 fixed four *fatal* defects and was **necessary but nowhere near sufficient** — the
Phase-1 end state still produced FID 206 in the pilot. Four further defects had to be found
in Phase 2, each proven by an executed reproduction:

| defect | evidence | effect |
|---|---|---|
| **σ_max = 1.0 was ~42× too small** for a valid VE prior (Song technique-1: max pairwise data distance = **41.88**) | pilot A→B: **206.49 → 116.87** | sampler initialised outside the prior |
| **DSM objective unweighted** — target −ε/σ makes the loss scale ∝ 1/σ², a 1.7e7 ratio over [0.01,42] | gradient collapsed onto the smallest σ | large-σ regime — where the trajectory *starts* — barely learned |
| **preconditioning h = x/σ** — network input scale spans **70×** across σ | pilot B→C: **116.87 → 74.70** | worst exactly at the low-σ end, where quality is decided |
| **EMA had no bias correction** — shadow seeded with the *random init*, decayed at 0.9999 | epoch 10: RAW **99.75** vs EMA **373.07** | **80%** of the epoch-10 "EMA" weights were still random init |

The EMA defect deserves emphasis: it would have produced a FID-vs-epoch curve measuring
*how fast the EMA forgets its initialisation*, not how fast the model learns — and it would
have looked like a plausible learning curve.

---

## Limitations (stated plainly)

1. **The baseline is undertrained.** FID was still falling at epoch 100. The 19.71 figure
   is a floor on what this architecture can do, not its converged value.
2. **FID=19.71 requires NFE=500.** At the NFE=100 the ITS framework typically uses, FID is
   52.96. Any controlled-vs-baseline comparison must be made **at equal NFE**.
3. **The EMA weights on disk are unrecoverable.** `ExponentialMovingAverage` is now fixed
   (warmup schedule, verified: old 80% init → new 0.0000% at epoch 10), but the shadow
   already saved in these checkpoints cannot be de-contaminated post hoc — it would require
   θ_init, which is not saved. **This run's gate is therefore evaluated on raw weights**,
   which are simply the trained model and perfectly valid. A future run gets a working EMA
   and should improve further.
4. **These FIDs are not comparable to the project's pre-audit numbers** — beyond the void
   sampler, FID itself was computed on *inverted* images (`normalize=True` fed uint8), so
   the old numbers are in a different pixel space.
5. **Model capacity is small** (1.61M params, `base_channels=64`, `channel_mults=(1,2,2)`).

## Reproduce

```bash
python scripts/launch_segmented_training.py \
    --script scripts/train_score_corrected_baseline.py \
    --target-epochs 100 --checkpoint-dir checkpoints/fmnist_score_corrected_baseline \
    --max-segment-hours 1.5
python scripts/run_gate_2_2.py
```

Wall clock: **2.1 h** for 100 epochs on a GTX 1650 Ti (4 GB), FP32, batch 256 — versus the
**176 h** run 5A consumed on CPU. The segmented launcher handed off cleanly mid-run
(segment 1 exited code 42 after 1.51 h at epoch 73; segment 2 auto-resumed from
`score_last.pt`), so the fault-tolerance was exercised live rather than assumed.
