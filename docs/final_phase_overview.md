# ITS Final Phase Overview

## Goals
- Train score-based generative models (U-Nets) on vision datasets via denoising score matching.
- Embed physics-informed diagnostics (entropy production, fluctuation theorems) alongside ML metrics.
- Couple learned scores with control policies inside variance-preserving SDE samplers.

## Pipeline
1. **Dataset loader** (its.data): TorchVision datasets with augmentation + normalisation.
2. **Score model** (its.models): Configurable U-Net backbone for multi-scale residual processing.
3. **Training loop** (its.training.score_training): Log-uniform noise schedule, EMA, gradient clipping.
4. **SDE sampler** (its.sde.score_sde): Variance-preserving discretisation with optional control corrections.
5. **Physics module** (its.physics): Jarzynski/Crooks/entropy estimators for post-analysis.

## Usage
- python -m scripts.train_score_model — trains the score network.
- python -m scripts.run_final_sampler --grid-output data/results/final_samples.png --overrides "experiment.sampler.num_steps=200" — generates samples (optionally with controller).

## Next Up
- Integrate FID/IS and thermodynamic inequality checks in evaluation reports.
- Extend controllers to operate directly in image space with shared weights.
- Add automated sweeps, logging (W&B/TensorBoard), and checkpoint orchestration.
