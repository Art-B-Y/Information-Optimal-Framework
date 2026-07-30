# ITS — Reproduction Guide

Exact shell commands to reproduce every experiment in the ITS project.
All commands must be run from the **project root directory**.
Activate the virtual environment first: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows).

---

## 0. Validate configuration files

Before running experiments, confirm all Hydra configs resolve correctly:

```bash
python scripts/validate_configs.py
```

Expected output: all configs print `[OK]` with no `[FAIL]` entries.

---

## 1. Toy experiment (double-well potential, CPU, ~30 s)

```bash
python scripts/run_experiment.py --config-name config experiment=double_well device=cpu seed=42
```

Produces: terminal loss/occupancy logs. No checkpoint is saved (toy experiments are diagnostic only).

For the Gaussian mixture potential:

```bash
python scripts/run_experiment.py --config-name config experiment=gaussian_mixture device=cpu seed=42
```

---

## 2. Migrate existing score checkpoints (one-time, ~5 min)

Run this once on the pre-Phase-1 checkpoints to add `model_config` and canonical key names:

```bash
python scripts/migrate_checkpoint.py --dir checkpoints/score
```

Verify the migration succeeded:

```bash
python -c "
import torch
s = torch.load('checkpoints/score/score_epoch_0050.pt', map_location='cpu', weights_only=False)
assert 'model_config' in s and 'model_state_dict' in s
print('Migration OK:', s['model_config'])
"
```

---

## 3. Baseline score training on FashionMNIST (3 epochs, CPU, ~10 min)

```bash
python -c "
from its.training.score_training import ScoreTrainingConfig, train_score_model
from its.data import DatasetConfig
from its.models import ScoreUNetConfig
cfg = ScoreTrainingConfig(
    epochs=3, lr=2e-4, device='cpu', log_interval=200,
    dataset=DatasetConfig(name='fashionmnist', batch_size=64, num_workers=0),
    model=ScoreUNetConfig(in_channels=1, base_channels=32, channel_mults=(1,2,2)),
    ema_decay=0.999, checkpoint_dir='checkpoints/score_fmnist', save_interval=1,
)
print(train_score_model(cfg))
"
```

For CIFAR-10 with the existing base_channels=128 architecture (GPU recommended):

```bash
python scripts/train_score_model.py \
    --config-name train_long_run \
    "experiment.score_training.epochs=10" \
    "experiment.score_training.checkpoint_dir=checkpoints/score_new" \
    "device=cuda"
```

---

## 4. Controlled training on FashionMNIST (3 epochs, CPU, ~30 min)

```bash
python scripts/train_controlled_score.py \
    --config-name controlled_mnist \
    "experiment.training.epochs=3" \
    "experiment.training.batch_size=32" \
    "experiment.training.path_kl_weight=0.1" \
    "experiment.training.control_weight=0.01" \
    "experiment.training.checkpoint_dir=checkpoints/controlled_fmnist" \
    "experiment.training.save_interval=1" \
    "device=cpu"
```

### 4a. Freeze pretrained score backbone (Session 4)

Load a pretrained CIFAR-10 score backbone and train **only** the control policy:

```bash
python scripts/train_controlled_score.py \
    --config-name controlled_cifar \
    "experiment.training.score_backbone_ckpt=checkpoints/score/score_epoch_0050.pt" \
    "experiment.training.freeze_score_model=true" \
    "experiment.training.epochs=2" \
    "experiment.training.checkpoint_dir=checkpoints/controlled_cifar_frozen" \
    "experiment.training.save_interval=1" \
    "device=cpu"
```

---

## 5. Hyperparameter sweep on FashionMNIST (Session 4, CPU, ~30-90 min)

Runs 6 configurations (A–F) of `control_weight`, `path_kl_weight`, `quality_weight`
with 1 epoch each on a 2000-sample subset:

```bash
python scripts/run_hyperparam_sweep.py
```

Results saved to `data/results/hyperparam_sweep_fmnist.json`.

To run a subset of configs (e.g. A, C, E):

```bash
python scripts/run_hyperparam_sweep.py --configs A C E
```

---

## 6. Evaluation of controlled vs. baseline

Controlled SDE (requires both checkpoints):

```bash
python scripts/eval_samples.py \
    --config-name controlled_mnist \
    --mode controlled \
    --model-ckpt checkpoints/controlled_fmnist/controlled_last.pt \
    --control-ckpt checkpoints/controlled_fmnist/controlled_last.pt \
    --quick
```

With reproducible seed (Session 4, Step 6C):

```bash
python scripts/eval_samples.py \
    --config-name controlled_mnist \
    --mode controlled \
    --model-ckpt checkpoints/controlled_fmnist/controlled_last.pt \
    --control-ckpt checkpoints/controlled_fmnist/controlled_last.pt \
    --eval-seed 42 \
    --quick
```

Full publication-quality evaluation with dual-seed FID variance (Session 4, Steps 6D/6E):

```bash
python scripts/eval_samples.py \
    --config-name controlled_mnist \
    --mode controlled \
    --model-ckpt checkpoints/controlled_fmnist/controlled_last.pt \
    --control-ckpt checkpoints/controlled_fmnist/controlled_last.pt \
    --eval-seed 42 \
    --full-eval
```

DDPM baseline (uses score model checkpoint only):

```bash
python scripts/eval_samples.py \
    --config-name controlled_mnist \
    --mode ddpm \
    --model-ckpt checkpoints/score_fmnist/score_epoch_0003.pt \
    --baseline-only \
    --quick
```

DDIM baseline:

```bash
python scripts/eval_samples.py \
    --config-name controlled_mnist \
    --mode ddim \
    --model-ckpt checkpoints/score_fmnist/score_epoch_0003.pt \
    --baseline-only \
    --quick
```

---

## 7. IPF (Schrödinger Bridge) alternating training (Session 4)

Quick smoke test — 2 iterations, 1 epoch/phase, 500-sample FashionMNIST subset:

```bash
python scripts/train_schrodinger_bridge.py \
    --ipf-iterations 2 \
    --epochs-per-phase 1 \
    --subset 500 \
    --batch-size 32 \
    --steps 5 \
    --base-channels 8 \
    --checkpoint-dir checkpoints/ipf_test \
    --jsonl-log logs/ipf_test.jsonl
```

Full training with evaluation grids:

```bash
python scripts/train_schrodinger_bridge.py \
    --ipf-iterations 5 \
    --epochs-per-phase 2 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/ipf_fmnist \
    --jsonl-log logs/ipf_fmnist.jsonl \
    --evaluate
```

---

## 8. Full benchmark comparison

```bash
python scripts/run_benchmark.py
```

Results are saved to `data/results/benchmark_table.json` and `data/results/benchmark_table.csv`.

---

## 9. Generate sample grids from the CIFAR-10 score checkpoint

```bash
python scripts/run_final_sampler.py \
    --config-name final_phase \
    --checkpoint checkpoints/score/score_epoch_0050.pt \
    --batch 64 \
    --grid-output data/results/samples_baseline_sde.png
```

---

## 10. Collect and plot training metrics

Parse training logs:

```bash
python scripts/collect_metrics.py \
    --type score \
    --logs logs/ \
    --output-csv data/results/score_metrics.csv \
    --output-json data/results/score_metrics.json
```

Validate log directories exist before collection:

```bash
python scripts/collect_metrics.py --type score --check
```

Plot metrics:

```bash
python scripts/plot_metrics.py --type score --csv data/results/score_metrics.csv
```

---

## 11. Monitor live training

```bash
python scripts/watch_training.py logs/controlled_train_XXXXXX.jsonl
```

Or with a single-pass (for CI/scripted use):

```bash
python scripts/watch_training.py logs/controlled_train_XXXXXX.jsonl --once
```

---

---

## 12. Session 6: GPU-accelerated training (GTX 1650 Ti, CUDA 12.1)

### 12a. Extended score training — FashionMNIST v2 (50 epochs, GPU, ~65 min)

```bash
python scripts/train_fmnist_score_v2.py --epochs 50 --batch-size 256
```

Checkpoints -> `checkpoints/score_fmnist_v2/score_epoch_00{05,10,...,50}.pt`
JSONL log -> `logs/fmnist_score_v2.jsonl` (includes `val_dsm_loss` every 5 epochs, LR per step)

### 12b. Extended score training — CIFAR-10 v2 (50 epochs, GPU, ~3-4 hrs)

```bash
python scripts/train_cifar10_score_v2.py --epochs 50 --batch-size 64 --base-channels 64
```

**Note:** Run AFTER FashionMNIST v2 to avoid GPU OOM on GTX 1650 Ti (4GB).

### 12c. Baseline evaluations — DDPM/DDIM/SDE at multiple NFE

```bash
python scripts/run_baseline_evals.py \
    --dataset fashionmnist \
    --model-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt \
    --nfe-list 50 100 200 500 \
    --num-samples 5000 \
    --seed 42
```

Results -> `data/results/baseline_evals_fashionmnist.json`

### 12d. Controlled Config D training (30 epochs, frozen backbone, 3 seeds)

```bash
python scripts/train_controlled_config_d.py \
    --score-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt \
    --epochs 30 --all-seeds
```

Results -> `data/results/controlled_config_d_results.json`

### 12e. 7-ablation study (6 configs, 5 epochs each)

```bash
python scripts/run_ablation_study.py \
    --score-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt \
    --epochs 5
```

Results -> `data/results/ablation_study.json`

### 12f. IPF toy convergence verification

```bash
python scripts/run_ipf_toy_convergence.py
```

Results -> `data/results/ipf_toy_convergence.json`

### 12g. Crooks/Jarzynski verification (Gaussian protocol)

```bash
python scripts/run_crooks_verification.py
```

Results -> `data/results/crooks_verification.json`

### 12h. Regenerate analytics report with v2 data

```bash
python scripts/generate_report.py --output data/results/its_analytics_report.html
python scripts/generate_paper_figures.py
python scripts/generate_results_table.py
```

---

## Notes

- CPU training is supported for all experiments but slow; use `device=cuda` whenever available.
- FashionMNIST and CIFAR-10 data are downloaded automatically to `./data/tensors/`.
- All checkpoints include `model_config` so architecture is always recovered at load time.
- `--quick` in `eval_samples.py` overrides sample count to 256 for fast development iteration.
- `--eval-seed N` makes FID/IS reproducible; `--full-eval` uses full sample counts and reports dual-seed FID variance.
- Default evaluation sample counts: 5000 for FashionMNIST, 10000 for CIFAR-10.
- FID is always computed against the **test split** (`train=False`) to avoid inflated scores from memorised training images.
- The `DatasetConfig.subset_size` and `ControlledScoreTrainingConfig.dataset_subset_size` fields enable fast sweep pilots on CPU.
