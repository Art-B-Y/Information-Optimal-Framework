# ITS — Architecture Reference

This document describes the mathematical formulation and software architecture of the
Information-Theoretic Sampler (ITS) framework.  It is intended for readers who want
to understand the method at the level of a methods section.

---

## 1. Controlled VP-SDE

The ITS framework models generation as a *controlled* variance-preserving SDE:

```
dX_t = [f(X_t, t) + g(t)² s_θ(X_t, σ_t)] dt
      + u_θ(X_t, t) dt
      + g(t) dW_t
```

where:

- `f(x, t) = -½ β(t) x` — VP drift
- `g(t) = √β(t)` — diffusion coefficient
- `β(t) = β_min + t(β_max - β_min)` — linear noise schedule
- `s_θ(x, σ)` — score model (denoising score matching)
- `u_θ(x, t)` — control policy (learned correction)
- `σ_t = σ_min (σ_max / σ_min)^t` — log-uniform sigma schedule

---

## 2. Denoising Score Matching (DSM) objective

The score model is trained with the DSM loss:

```
L_DSM = E[‖s_θ(x + σ ε, σ) − (−ε / σ)‖²]
```

where `ε ~ N(0, I)` and `σ` is drawn log-uniformly from `[σ_min, σ_max]`.

---

## 3. Control objective

The joint training objective combines four terms:

```
L = L_DSM
  + λ_ctrl · (1/T) Σ_t ‖u_θ(X_t, t)‖² Δt        (control energy)
  + λ_KL   · (−1/N Σ_i log_RN_i)                  (path-KL / Girsanov)
  + λ_qual · ‖μ_gen − μ_real‖²                     (feature matching)
```

**Path-KL (Girsanov log-RN):**

```
log(dQ/dP)|_path = Σ_t [ u · ε √(β Δt) − ½ ‖u‖² Δt ]
```

where `Q` is the controlled measure and `P` is the reference SDE measure.

**Jarzynski estimator:**

The free-energy difference ΔF is estimated via:

```
ΔF ≈ −(1/β) LogMeanExp(−β W)
   = −(1/β) [logsumexp(−β W_i) − log N]
```

where `W_i` is the per-sample work (negative log-RN) and the logsumexp is
numerically stabilised via max-shift.

---

## 4. Score model: `ScoreUNet`

The score model is a lightweight U-Net in `src/its/models/score_unet.py`:

```
Input (B, C, H, W)  →  h = x / σ
Down path:  [DoubleConv → GroupNorm → SiLU → Conv3×3 + skip] × L levels
                          ↓ strided Conv2d (downsampler)
Bottleneck: [optional BottleneckSelfAttention]
Up path:    [ConvTranspose2d → cat(skip) → DoubleConv] × (L−1) levels
Output:     Conv1×1 → score = output / σ
```

`DoubleConv` is a residual block (GN + SiLU + Conv3×3 × 2 with GroupNorm skip).

**Session 4 additions:**

- `use_time_embedding: bool = False` — when enabled, adds a `_SinusoidalTimeEmbedding`
  that maps `log(σ)` to a D-dimensional embedding, injected into each residual block
  via learned channel-wise affine modulation (FiLM: `x * (1 + scale) + shift`).
  Controlled by `ScoreUNetConfig.time_embed_dim` (default 128).

- `use_attention: bool = False` — when enabled, adds a `_BottleneckSelfAttention`
  module (4-head `nn.MultiheadAttention`, spatial flatten/reshape) after the final
  down block.  Controlled by `ScoreUNetConfig.attention_heads` (default 4).

Both flags default to `False` for full backward compatibility with `score_epoch_0050.pt`.

---

## 5. Control policy

Two control policy architectures are provided:

### 5a. MLP `ControlPolicy` (toy / flat experiments)

Used for FashionMNIST / MNIST with `state_dim=784`:

```
Input: [x_flat ‖ t_embed]  →  MLP(depth×hidden_dim)  →  Linear(hidden_dim, state_dim)
```

`t_embed = Tanh(Linear(1, time_embedding_dim))`.

### 5b. `ConvControlPolicy` (image experiments)

Preferred for spatial structure:

```
Input (B, C, H, W)  →  Conv1×1  →  [ResBlock × N]  →  Conv1×1  →  Output (B, C, H, W)
```

Each `ResBlock` uses FiLM conditioning via `SinusoidalTimeEmbedding`.
Output projection is **zero-initialised** so the controller starts as a near-zero
perturbation to the SDE drift.

---

## 6. Schrödinger Bridge / IPF

`SchrodingerBridgeSampler` in `src/its/samplers/sb_solver.py` implements
Iterative Proportional Fitting (IPF) for Schrödinger bridges.

**Architecture:**

```
forward_policy  (u_θ, ConvControlPolicy): drives X_0 → X_T
backward_policy (v_φ, ConvControlPolicy): drives X_T → X_0
```

**IPF alternating loop** (see `scripts/train_schrodinger_bridge.py`):

```
for ipf_iteration in range(N):
    phase = "forward":  freeze v_φ, minimise L_IPF(u_θ) for E epochs
    phase = "backward": freeze u_θ, minimise L_IPF(v_φ) for E epochs
```

**IPF loss:**

```
L_IPF = (1 / 2σ²) Σ_t ‖u(X_t, t)‖² Δt   ≈   E_Q[log(dQ/dP)]
```

This is the Girsanov path-KL from the controlled measure `Q` to Brownian motion `P`.

**Euler-Maruyama integrator:**

```
X_{n+1} = X_n + σ u(X_n, t_n) dt + σ √dt ε_n,   ε_n ~ N(0, I)
```

Forward: `t_n = n / (steps − 1)` increasing from 0 to 1.
Backward: `t_n = 1 − n / (steps − 1)` decreasing from 1 to 0.

Girsanov log-weight per step: `u · dW − ½ ‖u‖² dt`.

---

## 7. Evaluation pipeline

`src/its/eval/evaluator.py` computes FID and IS using `torchmetrics`.

**Critical correctness note:** FID is always computed against the **test split**
(`train=False`).  Using the training split inflates FID in favour of models that
memorise training data (fixed in Session 4 audit, CAT1-1/CAT4-1).

**Session 4 hardening (Step 6):**

| Flag | Effect |
|------|--------|
| `num_samples < 2048` | `UserWarning` emitted (Step 6A) |
| `EvaluationConfig.seed` | Sets RNG seed for reproducible FID/IS (Step 6C) |
| `EvaluationConfig.full_eval=True` | Overrides to 5000/10000 samples (Step 6D) |
| `full_eval=True` | Runs two seeds, reports `fid_mean` and `fid_std` (Step 6E) |

---

## 8. Configuration system

All experiments use [Hydra](https://hydra.cc/) with config files in `configs/`.

Key configs:

| File | Purpose |
|------|---------|
| `configs/experiment/controlled_mnist.yaml` | FashionMNIST controlled training |
| `configs/experiment/controlled_cifar.yaml` | CIFAR-10 controlled training |
| `configs/train_long_run.yaml` | Score-only training with `base_channels=128` |

New fields added in Session 4 (wired to both YAML files):

- `training.quality_weight` — feature-matching loss weight
- `training.nan_tolerance` — max consecutive NaN batches
- `training.save_every_n_epochs` — epoch-wise checkpoint interval
- `training.jsonl_log` — JSONL step log path
- `training.score_backbone_ckpt` — pretrained score model path
- `training.freeze_score_model` — freeze score model during training

New fields added in Session 6 (wired to both YAML files and direct training scripts):

- `training.use_lr_schedule` — enable cosine annealing LR schedule over total steps
- `training.lr_min` — minimum learning rate for cosine schedule (default 1e-6)
- `score_training.val_every_n_epochs` — compute validation DSM loss every N epochs
- `training.log_sample_diversity` — log mean pairwise L2 sample diversity metric

---

## 8b. Session 6 training improvements

### Cosine annealing LR schedule

Both `ScoreTrainingConfig` and `ControlledScoreTrainingConfig` now support:

```python
use_lr_schedule: bool = True
lr_min: float = 1e-6
```

When enabled, `CosineAnnealingLR` is applied over the full training horizon:
`T_max = epochs × batches_per_epoch`.  The current LR is logged to JSONL at every `log_interval` step.

### Validation loss monitoring

`ScoreTrainingConfig.val_every_n_epochs` (default 5): after each qualifying epoch,
DSM loss is computed on the test split and the overfitting ratio `val/train` is
logged.  A warning is emitted if `ratio > 1.2`.

### EMA resume fix

`_load_checkpoint` in `controlled_score_training.py` now restores the EMA state
dict (Issue S6-1 fixed).  Checkpoints before Session 6 will not restore EMA on
resume, but new checkpoints include `ema_state_dict`.

---

## 9. Module map

```
src/its/
├── __init__.py              — top-level exports
├── models/
│   └── score_unet.py        — ScoreUNet, ScoreUNetConfig
├── controllers/
│   └── neural_control.py    — ControlPolicy, ConvControlPolicy, configs
├── sde/
│   ├── score_sde.py         — ScoreSDEConfig, ScoreSDESimulator
│   └── conversions.py       — eps_to_score, score_to_eps
├── training/
│   ├── score_training.py    — ScoreTrainingConfig, train_score_model, EMA
│   └── controlled_score_training.py — ControlledScoreTrainingConfig, train_controlled_score
├── samplers/
│   ├── ddpm.py              — DDPMSampler, DDIMSampler
│   └── sb_solver.py         — SchrodingerBridgeSampler, SBSamplerConfig
├── eval/
│   └── evaluator.py         — EvaluationConfig, evaluate_sampler, evaluate_ddpm_baseline
├── data/
│   └── datasets.py          — DatasetConfig, build_dataset
└── physics/
    └── fluctuation.py       — jarzynski_work_estimate, entropy_proxy
```
