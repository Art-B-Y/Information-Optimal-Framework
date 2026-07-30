# ITS — Next Phase Roadmap

Tracks completed work and planned extensions.  Updated after Session 6.

---

## Completed (Sessions 1–4)

### Session 1–2: Core infrastructure + bug fixes
- ✓ Baseline VP-SDE score training loop (`ScoreTrainingConfig`, `train_score_model`)
- ✓ DSM loss with log-uniform sigma sampling
- ✓ EMA of score model weights (decay=0.999)
- ✓ Jarzynski/Crooks free-energy estimators with logsumexp stability
- ✓ Checkpoint save/load with architecture configs for reproducible rebuilds
- ✓ Bug 1: checkpoint architecture mismatch on load
- ✓ Bug 2: `ControlConfig.state_dim` mismatch with flattened image size
- ✓ Bug 3: sigma broadcasting (missing view)
- ✓ Bug 4: `eval_samples.py` checkpoint requirement not enforced
- ✓ Bug 5: `path_kl_weight=0` decoupled info-theoretic objective

### Session 3: Training hardening + new features
- ✓ CIFAR-10 50-epoch score backbone (`checkpoints/score/score_epoch_0050.pt`)
- ✓ Feature-matching quality loss via frozen Inception-v3
- ✓ `ConvControlPolicy` for spatial structure in image experiments
- ✓ NaN guard with `nan_tolerance` counter
- ✓ `controlled_last.pt` (latest-checkpoint pointer)
- ✓ JSONL step logging + `watch_training.py`
- ✓ Mixed-precision and gradient accumulation support
- ✓ `SchrodingerBridgeSampler` with IPF loss

### Session 4: Audit + architecture + evaluation hardening
- ✓ Full 43-file codebase audit (7 issues found and fixed)
  - `score_unet.py`: sigma clamp (NaN guard)
  - `conversions.py`: `eps_to_score` sigma clamp
  - `fluctuation.py`: dead `log_z` variable removed
  - `evaluator.py`: FID now uses test split (`train=False`)
  - YAML configs: 4 missing fields added
- ✓ `DatasetConfig.subset_size` + `ControlledScoreTrainingConfig.dataset_subset_size`
- ✓ Score backbone loading + `freeze_score_model` flag (Step 3)
- ✓ `ScoreUNet` architecture improvements (Step 4):
  - `SinusoidalTimeEmbedding` per-block injection (`use_time_embedding: bool = False`)
  - `BottleneckSelfAttention` (`use_attention: bool = False`)
  - Full backward compatibility with `score_epoch_0050.pt`
- ✓ `scripts/train_schrodinger_bridge.py` — IPF alternating training driver (Step 5)
- ✓ Evaluation pipeline hardening (Step 6):
  - 6A: `num_samples < 2048` warning
  - 6B: `train=False` for FID (audit fix)
  - 6C: `--eval-seed` flag
  - 6D: `--full-eval` flag
  - 6E: dual-seed FID variance estimate
- ✓ Test count: 17 → 21 (4 new tests for Steps 3/4)
- ✓ CPU hyperparameter sweep infrastructure (`scripts/run_hyperparam_sweep.py`)
- ✓ `docs/architecture.md` added

---

## Completed (Sessions 5–6)

### Session 5: Phase 5 hardening + extended evaluation
- ✓ Early stopping with `early_stop_patience` and `early_stop_min_delta`
- ✓ Reproducibility manifest (`reproducibility_manifest.json`) with seed verification
- ✓ 3-seed reliability assessment for controlled training
- ✓ Extended sweep (Config D 10-epoch + Config C/F comparison)
- ✓ Test count: 21 → 26 (5 new tests)
- ✓ Analytics HTML report with inline SVG charts (`generate_report.py`)
- ✓ Paper figures generation (`generate_paper_figures.py`)

### Session 6: GPU enablement + paper-grade results
- ✓ GPU setup: PyTorch reinstalled with CUDA 12.1 (torch 2.5.1+cu121, GTX 1650 Ti 4GB)
- ✓ GPU environment record: `data/results/gpu_environment.json`
- ✓ EMA resume fix (Issue S6-1): `_load_checkpoint` now restores EMA state dict
- ✓ `torch.load(weights_only=False)` fix (Issues S6-2, S6-3)
- ✓ Score training improvements (Step 7):
  - Cosine annealing LR schedule over total steps (`use_lr_schedule`, `lr_min`)
  - Validation DSM loss every N epochs (`val_every_n_epochs`), overfitting ratio warning
  - LR logged to JSONL every `log_interval` steps
- ✓ YAML configs updated with new fields (both controlled_mnist and controlled_cifar)
- ✓ FashionMNIST score v2: 50-epoch GPU training script (`train_fmnist_score_v2.py`)
- ✓ CIFAR-10 score v2: 50-epoch GPU training script (`train_cifar10_score_v2.py`)
- ✓ Baseline evaluations script: DDPM/DDIM/SDE at multiple NFE (`run_baseline_evals.py`)
- ✓ Controlled Config D training script: 3-seed reliability (`train_controlled_config_d.py`)
- ✓ Ablation study script: 6 configs (B/C/D/F/no-EMA/no-freeze) (`run_ablation_study.py`)
- ✓ IPF toy convergence verification (`run_ipf_toy_convergence.py`)
- ✓ Crooks/Jarzynski verification with harmonic protocol (`run_crooks_verification.py`)
- ✓ Control drift analysis from JSONL logs (`run_control_drift_analysis.py`)
- ✓ Report generator updated for JSONL format with val loss + LR traces
- ✓ generate_results_table.py updated to parse `baseline_evals_{dataset}.json`
- ✓ Test count: 26 → 34 (8 new tests, all passing)
- ✓ `docs/reproduction.md`: Section 12 added (GPU training commands)
- ✓ `docs/architecture.md`: Session 6 changes documented

---

## Near-term priorities (Phase 7)

### A. IPF training quality improvements (carry-over from Phase 5D)
- Use backward-policy trajectory as reference for forward training and vice-versa
  (proper IPF half-steps instead of zero-reference loss)
- Add endpoint matching loss: L2 or FID between `X_T` and target distribution
- Validate convergence on FashionMNIST with > 5 IPF iterations and 3+ epochs/phase

### B. Larger-scale controlled training
- Run 10-epoch controlled training on FashionMNIST with `path_kl_weight=0.1`
  and ConvControlPolicy (currently only MLP control is used in sweeps)
- Compare ConvControlPolicy vs MLP FID at equal parameter budget
- Profile and optimise `simulate_path()` — currently recomputes score at every step

### C. Architecture search
- Grid search over `use_time_embedding` × `use_attention` × `base_channels`
- Target: find configuration that improves FID by ≥ 5 points over baseline with < 2× wall-clock
- Log all runs to W&B for comparison

### D. Score distillation / consistency models
- Port `ScoreUNet` to output `x_0` directly (distillation target)
- Single-step generation from consistency model trained on top of epoch-50 backbone
- Measure NFE reduction: 50 steps → 1-4 steps

### E. Reproducibility and infrastructure
- Hydra multi-run sweeps integrated with `run_hyperparam_sweep.py`
- W&B logging for sweep results (replace local JSON)
- Docker image with CUDA 12.x base for fully reproducible CI
- Performance profiling: identify top-3 bottlenecks per training step

---

## Future (Phase 6+)

1. **FFHQ / CelebA-HQ** — scale `ScoreUNet` to 256×256 with attention at 16×16 and 8×8
2. **Perceptual losses** — LPIPS in feature-matching term, comparing to StableDiffusion CLIP embeddings
3. **Thermodynamic benchmarks** — FID/IS vs. control-energy Pareto front; compare with DDIM/DPM-Solver
4. **JAX port of controller** — Flax/Equinox for end-to-end differentiable simulation on GPU/TPU
5. **Containerisation** — Docker + CUDA base, deterministic environment, CI integration
