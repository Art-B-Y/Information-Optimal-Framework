# Information Track Sampler (ITS)

> ## ⛔ READ FIRST — a correctness audit voided all pre-2026-07-15 results
>
> A full-repository audit on **2026-07-15** established that **every experimental result
> this project produced before that date is void.** Not imprecise — void. Seven defects
> were found, each proven by an executed minimal reproduction:
>
> | # | Defect | Effect |
> |---|--------|--------|
> | A1 | Reverse-SDE drift had an inverted sign and a spurious factor 2 on the score | The sampler never converged to the data distribution (~84× the true variance, **flat FID across all NFE**) |
> | A2 | Score model trained in **VE** (`x + σ·ε`) but sampled with a **VP** sampler | Incompatible diffusion families |
> | A3 | `control_weight` defaults to 0 and *gates* the control term | **Control was silently disabled in every evaluation** — every "controlled" FID was the uncontrolled sampler's |
> | A4 | Girsanov path-KL wrong by a time-varying factor β_t | Every reported path-KL was wrong |
> | B  | `sigma_max=1.0` was ~42× too small for a valid VE prior | Sampler initialised outside the prior |
> | C  | DSM objective **unweighted** (loss scale ∝ 1/σ²) | Gradient collapsed onto the smallest σ |
> | D  | FID computed on **inverted** images (`normalize=True` fed uint8) | FIDs not comparable to published numbers |
>
> **Consequence for the project's headline claim.** The "controller collapse" finding
> (FID 327.47, Sessions 9–10) is substantially an **artifact of A3**: controlled and
> baseline FID were identical because they were *literally the same computation*. It
> must be re-established on the corrected pipeline before it means anything.
>
> **The pre-audit numbers you may find in old docs — FID 327.47, 236.28, "eval matrix
> done", "paper-grade" — are all void.** See
> [`data/results/RESULTS_ARE_VOID_READ_ME.md`](data/results/RESULTS_ARE_VOID_READ_ME.md),
> [`docs/current_state_diagnosis.md`](docs/current_state_diagnosis.md).
>
> **Authoritative state:** the corrected pipeline and the canonical baseline
> `checkpoints/fmnist_score_corrected_baseline/`. Validation status and the current
> baseline FID are in [`docs/gate_2_2_report.md`](docs/gate_2_2_report.md).
> Roadmap: [`docs/roadmap_current.md`](docs/roadmap_current.md).
>
> **The lesson worth carrying:** the test suite was **76/77 green throughout**, because
> every test asserted shapes, finiteness and importability — never that a number was
> scientifically *right*. Not one test would have failed if the sampler emitted `N(0,I)`.
> `tests/test_scientific_correctness.py` now holds checks that were each **falsified
> against the broken code** to prove they can fail.

## Corrected pipeline (authoritative)

```bash
# 1. Retrain the canonical baseline  (VE kernel <-> VE sampler, sigma in [0.01,42],
#    sigma^2-weighted DSM, EDM preconditioning, ~2h on a GTX 1650 Ti)
python scripts/launch_segmented_training.py \
    --script scripts/train_score_corrected_baseline.py \
    --target-epochs 100 --checkpoint-dir checkpoints/fmnist_score_corrected_baseline \
    --max-segment-hours 1.5

# 2. Gate 2.2 — the baseline MUST pass before any ITS number means anything
python scripts/run_gate_2_2.py

# 3. Controller (refuses to run unless Gate 2.2 passed)
python scripts/train_controlled_corrected.py
```

---

ITS now spans two tiers:

1. **Beta sandbox** — fast toy problems (double well / Gaussian mixture) for controller design.
2. **Final phase** — high-dimensional, dataset-driven score models with physics-aware diagnostics.

Both tiers share the same codebase so improvements carry forward automatically.

## What You Get
- Modular `its/` research package (controllers, samplers, score models, physics diagnostics).
- Hydra + Typer CLIs for both beta (`scripts/run_experiment.py`) and final-phase pipelines (`scripts/train_score_model.py`, `scripts/run_final_sampler.py`).
- Dataset tooling (CIFAR-10, MNIST, FashionMNIST) and score-based generative infrastructure.
- Thermodynamics helpers (entropy production, Crooks/Jarzynski estimators) to benchmark energy/information flow.

## Environment
All dependencies are range-pinned for cross-platform compatibility (`numpy 1.26.x`, `jax/jaxlib 0.4.x`, `torch 2.2-2.3`, `diffrax 0.5.x`, `torchmetrics`, `einops`, `scipy`, `tqdm`, etc.). Use Python 3.11 for the most reliable experience; Python 3.12 works if matching wheels are available.
```bash
# Create (optional) conda env with Python 3.11
conda env create -f env.yml
conda activate its-beta

# Or use plain pip (ensure Python >=3.11,<3.13)
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quickstart
```bash
# Run default double-well (beta)
python -m scripts.run_experiment

# Switch to Gaussian mixture toy
python -m scripts.run_experiment --config-name config --overrides experiment=gaussian_mixture

# Train score model on CIFAR-10 (final phase)
python -m scripts.train_score_model --config-name train_long_run  # checkpoints in checkpoints/score

# Sample from score-based ITS (uses configs/final_phase.yaml by default)
python -m scripts.run_final_sampler --batch 32 --output samples.pt --grid-output samples.png
```
Hydra overrides work as usual (e.g. `-o experiment.score_training.device=cpu -o sampler.config.num_steps=200` or `--overrides "experiment.score_training.device=cpu sampler.config.num_steps=200"`).

## Project Layout
```
src/its/
  controllers/   # Torch policies and helpers
  data/          # TorchVision dataset tooling
  models/        # Score networks (U-Net variants)
  samplers/      # Controlled SDE simulator (beta)
  sde/           # Score-based SDE samplers (final phase)
  objectives/    # Energy & quality objectives
  training/      # Controller and score-model training loops
  experiments/   # Beta + final-phase bundles
  metrics/       # Path summaries and diagnostics
  physics/       # Entropy / fluctuation-theorem estimators
  utils/         # Logging, seeding, interop helpers
configs/         # Hydra configs for experiments
scripts/         # CLI entry points
notebooks/       # Exploratory analysis (existing work retained)
docs/            # Design notes (see beta_phase_blueprint.md)
```

## Testing & QA
```bash
pytest          # runs sampler + training smoke tests
ruff check      # optional linting
```

## Diagnostics & Outputs
- **Beta tier** logs control energy, entropy proxy, occupancy balance, variance, etc., to verify no single-well collapse and bounded energy use.
- **Final tier** adds score-matching losses, batch symmetrisation, Jarzynski/Crooks estimators, and dataset-aware metrics ready for FID/Frechet integration.

Watch for:
- Occupancy balance ≈ 0 (beta) – ensures symmetric well occupancy.
- Control energy drifting down once loss plateaus – indicates efficient steering.
- Score-training loss stability – points to converging denoising score matching.

## Roadmap
- Attach real evaluation metrics (FID/IS, energy budgets, entropy production bounds).
- Introduce controller-in-the-loop training for high-dimensional samplers.
- Add logging + checkpointing (W&B / TensorBoard) and Dockerised reproducibility.


> **PowerShell tip:** line continuation uses backticks. Prefer single-line overrides, e.g.
> `python -m scripts.train_score_model --overrides "experiment.score_training.resume_from=checkpoints/score/score_epoch_0010.pt score_training.epochs=50 score_training.model.base_channels=128 score_training.dataset.batch_size=128"`


> Example (PowerShell): `python -m scripts.train_score_model --overrides "experiment.score_training.resume_from=checkpoints/score/score_epoch_0010.pt experiment.score_training.epochs=50 experiment.model.base_channels=128 experiment.score_training.dataset.batch_size=128"`


> Tip: advanced changes can still use `-o` / `--overrides` if needed.

> For custom runs: copy `configs/train_long_run.yaml`, tweak the values, then run `--config-name <your_file>`.
