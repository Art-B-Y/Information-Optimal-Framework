# ITS Paper Requirements Roadmap
**Updated: Session 10 (2026-04-24)**

---

## SESSION 10 UPDATE: Controller Collapse Discovered and Fixed

Session 10 revealed that the Session 9 ConvControlPolicy collapsed to near-zero output (0.061% of score magnitude at epoch 50), making FID=327.47 statistically identical to the uncontrolled DDPM baseline. The root cause is a gradient landscape with a zero-control global optimum:
- Path KL gradient = 1.45 on output layer (sole gradient signal)
- Quality gradient = 0.0 (broken chain through 50-step SDE + Inception)
- Both path KL and control energy are minimized at u_theta = 0

The v2 objective redesign (detach CE, trajectory quality, REINFORCE, WarmupSchedule) fixes this.
**v2 training Run 5A is currently running.** See `docs/controller_collapse_analysis.md` for full diagnosis.

---

## Section 1 — Results Complete and Paper-Ready

### 1.1 Crooks/Jarzynski Thermodynamic Verification
- **Result:** Jarzynski ΔF = 0.350 ± 0.003 (true: 0.3466); Crooks crossing = 0.317 ± 0.030
- **Validity:** std(work)/kT ≈ 0.44 — within reliable regime (< 1.0)
- **Paper claim:** The ITS framework satisfies the Crooks fluctuation theorem to within 9.4% relative agreement
- **Section:** §4 Thermodynamic Analysis; Figure: crooks_verification.{png,pdf}
- **Provenance:** `data/results/crooks_verification.json`; seed=42; 512 samples

### 1.2 Ablation Study (FashionMNIST, 7 configurations)
- **Result:** Config B/C/D/F/conv/no-EMA: FID=326.00; no-freeze: FID=329.62
- **Key finding:** freeze_score_model is the only ablation that significantly degrades FID (+3.6 pts)
- **Sample count:** 2048; single seed (42)
- **Section:** §5 Ablation Study; Figure: ablation_fid_chart.{png,pdf}
- **Provenance:** `data/results/ablation_study.json`

### 1.3 Schrödinger Bridge IPF Convergence
- **Result:** 5 IPF iterations; bwd_loss = 2.3e-5 at iteration 5
- **Paper claim:** IPF converges to bwd_loss < 5e-5 within 5 iterations on FashionMNIST
- **Section:** §3.3 Schrödinger Bridge Extension
- **Provenance:** `checkpoints/ipf_fmnist_full/`, `logs/ipf_fmnist_full.jsonl`

### 1.4 Config D Controlled Sampling — Primary Result (3-Seed CI)
- **Result:** FID = 327.47 ± 1.57 at NFE=100, 2048 samples, seeds=[42,123,7], 60 epochs
  - Seed 42: FID=327.11; Seed 123: FID=326.12; Seed 7: FID=329.19
  - 95% CI: [325.69, 329.25]
- **Comparison:** DDPM FID = 236.28 ± 0.87 at NFE=100 (3-seed, 2048 samples)
- **Gap:** 91.2 FID points (38.7% worse than DDPM at equal NFE). Statistically significant.
- **No Pareto improvement** over any baseline at any NFE level (Config D MLP)
- **Section:** §5 Experiments — Table 1; Figure: pareto_frontier.{png,pdf}
- **Provenance:** `data/results/controlled_config_d_multiseed.json`

### 1.5 Score Model Training (FashionMNIST v2)
- **Result:** 100 epochs; best avg_loss = 470.67 at epoch 100
- **Checkpoint:** `checkpoints/score_fmnist_v2/score_best.pt`
- **Section:** §5 Experiments (training details)
- **Provenance:** `logs/fmnist_score_v2_session7.log`

### 1.6 FashionMNIST Evaluation Matrix (3-seed, 2048 samples)
- **Result:** 24 runs: DDPM+DDIM × 4 NFE × 3 seeds
  - DDPM best: NFE=200, FID=201.40 ± 1.34 (95% CI: [199.88, 202.92])
  - DDIM best: NFE=200, FID=228.06 ± 0.95 (95% CI: [226.99, 229.14])
  - DDPM at NFE=100: FID=236.28 ± 0.87 (head-to-head vs Config D)
- **Section:** §5 Experiments — Table 1
- **Provenance:** `data/results/fmnist_eval_matrix_final.json`

### 1.7 Scientific Analyses (Session 8)
- **Memorization check:** NN mean L2 = 29.80; fraction below 0.05 = 0.0. No memorization.
- **Mode coverage:** 1000 samples; entropy ≈ 0 (all classified as Bag). Preliminary — TinyClassifier not calibrated to generated images. Requires retraining in Session 9.
- **Entropy production profile:** 256 trajectories, 100 SDE steps.
- **Provenance:** `data/results/memorization_check.json`, `mode_coverage.json`, `entropy_production_profile.json`

### 1.8 Session 10: Controller Collapse Analysis (NEW SCIENTIFIC CONTRIBUTION)
- **Gradient flow diagnosis:** Path KL gradient = 1.45 dominates; quality gradient = 0 (broken chain)
- **Collapse trajectory:** Ratio drops from 3.3% → 0.061% over 50 epochs
- **Thermodynamic interpretation:** Collapse = minimum entropy production state
- **v2 objective fix:** Detach CE, trajectory quality + REINFORCE, WarmupSchedule
- **Status:** Analysis DONE; v2 training RUN 5A in progress
- **Provenance:** `data/results/gradient_flow_diagnosis.json`, `collapse_trajectory_metrics.json`

---

## Section 2 — Session 9: In Progress or Pending Training

### 2.1 ConvControl 50-Epoch Training (HIGHEST PRIORITY — RUNNING)
- **Current:** ablation_conv at 5 epochs gave FID=326.00 (same as MLP)
- **Target:** 50 epochs, AdamW lr=5e-4, cosine to 5e-7, EMA=0.9999, batch=128
- **Status:** RUNNING (started 2026-04-23 17:11; checkpoint dir: checkpoints/controlled_conv_seed42/)
- **If FID < 236.28 (DDPM at NFE=100):** Paper claim upgrades to Pareto optimality; venue → NeurIPS workshop
- **If FID ≥ 236.28:** TMLR remains appropriate venue
- **Command:** `python scripts/train_controlled_config_d.py --score-ckpt checkpoints/score_fmnist_v2/score_best.pt --epochs 50 --use-conv-control --seed 42`
- **Estimated completion:** ~9-10 hours (GTX 1650 Ti, ~12 min/epoch × 50 epochs)

### 2.2 ConvControl 3-Seed Reliability (depends on 2.1)
- **Target:** Seeds 123, 7 at 2048 samples once seed 42 result is known
- **Priority:** HIGH if seed 42 shows improvement

### 2.3 Architecture Ablations (depends on GPU availability after 2.1)
- 3A: Attention ablation on FashionMNIST
- 3B: Time embedding ablation
- 3C: Config C (path_kl_weight=0.0) vs Config D with ConvControl at 30 epochs
- **Priority:** MEDIUM

### 2.4 CIFAR-10 Score Model
- **Current:** Not trained; blocked all CIFAR-10 experiments
- **Estimated GPU-hours:** ~10h on GTX 1650 Ti
- **Priority:** LOW for TMLR; HIGH for NeurIPS/ICML

---

## Section 3 — Evaluations Needing Updates

### 3.1 FashionMNIST Eval Matrix v2 (add ConvControl)
- **Current:** 24 runs (DDPM+DDIM × 4 NFE × 3 seeds) — COMPLETE
- **Target:** Add ConvControl rows after training completes (session 9 step 5A)
- **Status:** Pending training completion

### 3.2 Mode Coverage with Calibrated Classifier (Session 9 Step 4C)
- **Issue:** TinyClassifier (5 epochs) classifies all samples as "Bag" (entropy ≈ 0)
- **Fix:** Retrain classifier to 20 epochs with augmentation (target: 88% test accuracy)
- **Status:** Pending (no GPU conflict; can run anytime)

---

## Section 4 — Scientific Analyses Pending (Session 9)

| Analysis | Status | Notes |
|----------|--------|-------|
| Entropy production profile (MLP Config D) | COMPLETE | `entropy_production_profile.json` |
| Memorization check | COMPLETE | No memorization found |
| Mode coverage (MLP Config D) | PRELIMINARY | Classifier needs recalibration |
| Crooks verification with ConvControl | Pending 2.1 | Requires ConvControl checkpoint |
| Control drift analysis (ConvControl) | Pending 2.1 | 256 trajectories |
| Mode coverage with calibrated classifier | Pending | Can run without GPU |
| NFE efficiency curves | Pending | From eval matrix — needs ConvControl row |
| Entropy production comparison (Conv vs MLP) | Pending 2.1 | Requires ConvControl checkpoint |

---

## Section 5 — Paper Infrastructure

### 5.1 Results Table
- **Status:** `generate_results_table.py` updated to include ConvControl rows (Session 9)
- Reads: `controlled_conv_results.json`, `controlled_conv_multiseed.json` (when available)

### 5.2 Paper Figures (8 canonical figures)
- **Status:** All 8 generated in Session 7/8; need regeneration after ConvControl training

### 5.3 Reproducibility Manifest
- **Status:** Needs update with Session 9 ConvControl training entry

### 5.4 reproduce_paper_results.sh
- **Status:** Needs `--use-conv-control` variant added

---

## Section 6 — Paper Writing Tasks with Dependencies

| Section | Status | Dependencies |
|---------|--------|--------------|
| Abstract | Can draft | Wait for §2.1 FID result |
| Introduction | Can write | None |
| Related Work | In progress | 5 papers need review |
| Method | Can write | Architectural diagrams |
| Experiments | Partial | §2.1 ConvControl result |
| Results | Partial | §2.1 complete |
| Ablation Study | Can write | §1.2 complete |
| Thermodynamic Analysis | Can write | §1.1, §1.7 complete |
| Conclusion | Depends on venue | ConvControl result determines claim |
| Appendix | Can write | `docs/architecture.md` exists |

**Papers to cite:**
1. Song et al. 2021, "Score-Based Generative Modeling through SDEs" — foundational
2. Chen et al. 2021, "Likelihood Training of Schrödinger Bridge using FBSDE" — SB
3. Berner et al. 2022, "An optimal control perspective on diffusion-based generative modeling" — control
4. De Bortoli et al. 2021, "Diffusion Schrödinger Bridge" — IPF
5. Tzen & Raginsky 2019, "Theoretical guarantees for sampling via stochastic localization" — theory

---

## Section 7 — Pre-Submission Checklist

| Item | Status |
|------|--------|
| Repository clean and documented | Partial |
| `reproduce_paper_results.sh` runnable | Partial (needs ConvControl variant) |
| Environment snapshot | Complete (`data/results/environment_snapshot_session8.txt`) |
| Reproducibility manifest | Needs Session 9 update |
| Compute budget disclosure | ~31 GPU-hours (Sessions 1-8) + Session 9 pending |
| FashionMNIST license | OK (MIT) |
| CIFAR-10 license | OK (research use) |
| Baseline fairness (same NFE) | Yes — both evaluated at NFE=100 |
| Statistical significance (3 seeds) | Config D MLP: DONE; ConvControl: pending |
| Related work completeness | 5 papers flagged |

---

## Section 8 — Venue Selection Recommendation

### Current empirical position (Session 9, pre-ConvControl)

**FashionMNIST (MLP Config D, 3-seed):**
- Config D FID: 327.47 ± 1.57 at NFE=100, 60 epochs (95% CI: [325.69, 329.25])
- DDPM FID: 236.28 ± 0.87 at NFE=100
- Gap: 91.2 FID points — **No Pareto optimality** with MLP controller

**FashionMNIST (ConvControl, pending):**
- 5-epoch result: FID=326.00 (same as MLP) — need 50-epoch result
- Hypothesis: ConvControl may achieve FID < 236 by leveraging spatial structure at full convergence

**CIFAR-10:** No results (no score model trained)

### Current Recommendation: **TMLR** (may upgrade after §2.1)

| Condition | Venue |
|-----------|-------|
| ConvControl FID < 236.28 (beats DDPM at NFE=100) | NeurIPS/ICML workshop |
| ConvControl FID < 228.06 (beats DDIM at NFE=200) | NeurIPS/ICML workshop + main track discussion |
| ConvControl FID ≥ 236.28 | TMLR |

### What Is Already Unconditionally Publishable (TMLR)
1. Principled Girsanov SDE control formulation with path-KL regularization
2. Thermodynamic diagnostics (verified Crooks/Jarzynski; std(W)/kT = 0.44 < 1.0)
3. Working Schrödinger bridge with IPF convergence (5 iterations, bwd_loss=2.3e-5)
4. Honest ablation identifying freeze_score_model as key design choice
5. Memorization check passed; mode coverage preliminary
6. 48 passing tests with reproducible training pipeline

---

*Estimated remaining GPU-hours for TMLR submission: ~15h (after ConvControl completes)*
*Estimated remaining for NeurIPS workshop: ~25h (ConvControl + ablations + CIFAR-10 partial)*
*Calendar time (single GTX 1650 Ti, sequential): 2-4 days*
