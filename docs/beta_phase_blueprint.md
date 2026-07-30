# ITS Beta Phase Blueprint

## Goals
- Establish a reproducible sandbox for controlled SDE generative sampling experiments.
- Couple JAX-based samplers with PyTorch controllers in a modular way.
- Provide toy problems, baseline pipelines, and metrics for energy/information trade-offs.

## Folder Layout
- `src/its/`
  - `configs/`: YAML configs for samplers, controllers, and experiments.
  - `controllers/`: PyTorch control policies, initialization utilities.
  - `samplers/`: JAX/Diffrax SDE definitions and integrators.
  - `objectives/`: Loss terms for sample quality, control energy, entropy proxies.
  - `experiments/`: Experiment orchestration (toy 1D/2D tasks, runners).
  - `metrics/`: Evaluation routines (entropy production estimates, Wasserstein/FID proxies).
  - `training/`: Training loops, interop utilities.
  - `utils/`: Shared helpers (logging, seeding, data transforms).
- `configs/`: Top-level experiment entry points (Hydra-compatible).
- `scripts/`: CLI wrappers for launching experiments and evaluations.
- `tests/`: Lightweight unit/integration tests covering critical flows.
- `notebooks/`: Exploratory analysis linked to the beta pipelines.
- `docs/`: Design notes, research logs, and future work.

## Key Components
- **Controlled SDE Module**: Wraps Diffrax solvers with control augmentation hooks and entropy tracking.
- **Control Policies**: Torch modules exported through TorchScript-friendly interface for future deployment.
- **Coupling Layer**: Zero-copy bridging JAX <-> Torch via NumPy interop and device guards.
- **Objective Suite**: Configurable mix of quality, energy, and information terms.
- **Experiment API**: Declarative configuration constructing sampler + controller + objective + schedule.

## Deliverables in Beta Phase
1. Two toy experiments (1D double-well, 2D Gaussian mixture) with reproducible configs.
2. End-to-end training loop with logging hooks and checkpointing stubs.
3. Metrics for control energy, sample variance, and entropy-change estimates.
4. CLI runner `python -m its.scripts.run_experiment --config-name=<exp>`.
5. PyTest smoke tests validating sampler-controller round trip.
6. Documentation covering architecture + quickstart.

## Tech Stack
- Python 3.11
- JAX + Diffrax for SDE integration
- PyTorch + Torchvision (future dataset integration)
- Optax for optimization in JAX experiments
- Hydra + OmegaConf for configuration management
- Rich + Typer for CLI UX
- PyTest + Hypothesis for tests where applicable

## Next Steps toward Final Phase
- ✓ Scale controllers to high-dimensional score models (see `its.models`, `its.sde`).
- ✓ Integrate datasets (CIFAR-10 pipeline embedded via `its.data`).
- ✓ Add physics-aware metrics (entropy/Jarzynski/Crooks in `its.physics`).
- TODO: Full FID/thermodynamic reporting dashboards and GPU-optimised sweep tooling.
