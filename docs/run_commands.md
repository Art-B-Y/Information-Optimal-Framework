# ITS Project — Complete Ordered Terminal Command Guide

**Run all commands from the project root directory.**

Every command below is written exactly as it should be typed. Commands within a section that depend on the previous command completing successfully are marked with "Requires the previous command to succeed."

---

## Section 1 — Environment Verification

Confirm your Python version is 3.11 or 3.12 before doing anything else; the project's `pyproject.toml` enforces `>=3.11,<3.13`.

```
python --version
```

Confirm PyTorch is installed and check its version; the project requires `>=2.2,<2.4`.

```
python -c "import torch; print(torch.__version__)"
```

Confirm CUDA is available to PyTorch; if this prints `False` all training will run on CPU and will be extremely slow.

```
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Print the GPU name so you know which device training will use.

```
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Print total VRAM in GB so you can judge whether the default batch sizes will fit.

```
python -c "import torch; props = torch.cuda.get_device_properties(0); print(f'{props.total_memory/1e9:.1f} GB VRAM')" 
```

---

## Section 2 — Project Installation

Install the project package and all its dependencies in editable mode so that `import its` resolves correctly from any working directory. Requires Section 1 to confirm the Python environment is correct.

```
pip install -e .
```

Confirm the package installs cleanly by importing the top-level `its` module.

```
python -c "import its; print('its package OK')"
```

---

## Section 3 — Directory Setup

Create every directory that must exist before training or evaluation starts. Using `mkdir -p` (or the PowerShell equivalent shown below) means these commands succeed even if the directories already exist.

Create all checkpoint directories:

```
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['checkpoints/score_fmnist_v2','checkpoints/score_cifar10_v2','checkpoints/score','checkpoints/controlled_v2_5a_seed42','checkpoints/controlled_v2_5b_joint','checkpoints/controlled_v2_5c_nopkL','checkpoints/controlled_config_d_seed42','checkpoints/cifar10_controlled_d_seed42','checkpoints/ipf_fmnist']]; print('Checkpoint dirs created.')"
```

Create all log directories:

```
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['logs','logs/ipf','data/logs']]; print('Log dirs created.')"
```

Create all results and paper-figure directories:

```
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/results','data/paper_figures','data/tensors']]; print('Results dirs created.')"
```

---

## Section 4 — Config Validation

Run the Hydra config validator to confirm every experiment config resolves correctly before starting any training. Requires Section 2 (package installed) to succeed.

```
python scripts/validate_configs.py
```

---

## Section 5 — Checkpoint Migration

Migrate the existing CIFAR-10 score model checkpoint to add the `model_config` key and canonical key aliases. This must be run before any script that loads `checkpoints/score/score_epoch_0050.pt`. Requires Section 3 (checkpoint directories exist) to succeed. If this directory does not yet exist or the checkpoint has not been downloaded, skip this section and proceed to Section 7 from scratch instead.

```
python scripts/migrate_checkpoint.py --dir checkpoints/score
```

Verify the migration succeeded by dry-running it again; the output should say every checkpoint is already up-to-date.

```
python scripts/migrate_checkpoint.py --dir checkpoints/score --dry-run
```

---

## Section 6 — FashionMNIST Score Model Training

Train the FashionMNIST score model (100 epochs, base_channels=64, AMP enabled) using the segmented launcher with a 4-hour wall-clock limit per segment. The launcher restarts automatically from the latest checkpoint when a segment ends. Requires Sections 2–4 to succeed. **This command will run for multiple segments totalling approximately 8–16 hours on a GTX 1650 Ti.**

```
python scripts/launch_segmented_training.py --script scripts/train_fmnist_score_v2.py --run-id fmnist_score_v2 --target-epochs 100 --checkpoint-dir checkpoints/score_fmnist_v2 --max-segment-hours 4 --extra-args --epochs 100 --batch-size 256
```

After training finishes, verify that `score_best.pt` was saved (the eval scripts load this file).

```
python -c "from pathlib import Path; p=Path('checkpoints/score_fmnist_v2/score_best.pt'); print('score_best.pt EXISTS' if p.exists() else 'MISSING — check training logs')"
```

---

## Section 7 — CIFAR-10 Score Model Training

Resume CIFAR-10 score model training from the migrated checkpoint at epoch 50 and run to 200 epochs using the segmented launcher. Requires Section 5 (checkpoint migration) to succeed. **This command will run for many hours; the launcher will restart it automatically across segments.**

```
python scripts/launch_segmented_training.py --script scripts/train_cifar10_score_v2.py --run-id cifar10_score_v2 --target-epochs 200 --checkpoint-dir checkpoints/score_cifar10_v2 --max-segment-hours 4 --extra-args --epochs 200 --batch-size 64 --base-channels 64 --resume-from checkpoints/score/score_epoch_0050.pt
```

After training finishes, verify `score_best.pt` exists.

```
python -c "from pathlib import Path; p=Path('checkpoints/score_cifar10_v2/score_best.pt'); print('score_best.pt EXISTS' if p.exists() else 'MISSING — check training logs')"
```

---

## Section 8 — FashionMNIST Controlled Training (Session 10 v2 Objective)

Train the FashionMNIST controlled model using run 5A: frozen score backbone, v2 objective (trajectory quality + REINFORCE + detached control energy), ConvControlPolicy. Requires Section 6 (FashionMNIST score model) to succeed and `checkpoints/score_fmnist_v2/score_best.pt` to exist. **This run takes approximately 10–15 hours on a GTX 1650 Ti; the launcher handles all restarts.**

```
python scripts/launch_segmented_training.py --script scripts/train_redesigned_v2.py --run-id 5a --target-epochs 100 --checkpoint-dir checkpoints/controlled_v2_5a_seed42 --max-segment-hours 4 --extra-args --run 5a
```

Train run 5B (joint fine-tuning: score model unfrozen, distillation regularizer). Requires run 5A to complete before starting 5B (the 5B config loads the same score backbone independently, but sequentially is recommended to avoid GPU memory contention).

```
python scripts/launch_segmented_training.py --script scripts/train_redesigned_v2.py --run-id 5b --target-epochs 100 --checkpoint-dir checkpoints/controlled_v2_5b_joint --max-segment-hours 4 --extra-args --run 5b
```

Train run 5C (path-KL ablation: path_kl_weight=0, quality signal only). Requires Section 6 to succeed.

```
python scripts/launch_segmented_training.py --script scripts/train_redesigned_v2.py --run-id 5c --target-epochs 100 --checkpoint-dir checkpoints/controlled_v2_5c_nopkL --max-segment-hours 4 --extra-args --run 5c
```

---

## Section 9 — CIFAR-10 Controlled Training (Config D)

Train the CIFAR-10 controlled model (Config D: path_kl_weight=0.1, quality_weight=0.01, frozen backbone) for 50 epochs using the segmented launcher. Requires Section 7 (CIFAR-10 score model) to succeed and `checkpoints/score_cifar10_v2/score_best.pt` to exist. **This command can take 10–20 hours on a GTX 1650 Ti; the launcher handles all restarts.**

```
python scripts/launch_segmented_training.py --script scripts/train_cifar10_controlled_config_d.py --run-id cifar10_ctrl_d --target-epochs 50 --checkpoint-dir checkpoints/cifar10_controlled_d_seed42 --max-segment-hours 4 --extra-args --score-ckpt checkpoints/score_cifar10_v2/score_best.pt --epochs 50 --seed 42
```

---

## Section 10 — Schrödinger Bridge Training

Train the Schrödinger Bridge IPF model on FashionMNIST for 30 IPF iterations (2 epochs per forward/backward phase) using the segmented launcher. The launcher tracks progress via the `epoch` field saved in `ipf_last.pt`. Requires Section 3 (directories exist) to succeed. **30 iterations with the full dataset takes approximately 15–20 hours; the launcher handles restarts automatically.**

```
python scripts/launch_segmented_training.py --script scripts/train_schrodinger_bridge.py --run-id ipf_fmnist --target-epochs 30 --checkpoint-dir checkpoints/ipf_fmnist --max-segment-hours 4 --extra-args --ipf-iterations 30 --epochs-per-phase 2 --batch-size 32 --checkpoint-dir checkpoints/ipf_fmnist --jsonl-log logs/ipf/ipf_fmnist.jsonl --evaluate
```

---

## Section 11 — Evaluation

### 11A — FashionMNIST Baseline Samplers (DDPM, DDIM, SDE)

Evaluate DDPM, DDIM, and SDE samplers on FashionMNIST at NFE={50,100,200,500} with N=5000 samples and eval seed 42. Requires Section 6 (FashionMNIST score model) to succeed.

```
python scripts/run_baseline_evals.py --dataset fashionmnist --model-ckpt checkpoints/score_fmnist_v2/score_best.pt --num-samples 5000 --seed 42 --nfe-list 50 100 200 500 --samplers ddpm ddim sde
```

### 11B — FashionMNIST Controlled Sampler (ITS Run 5A, 3-seed)

Evaluate the v2 controlled sampler (run 5A) across three seeds to produce a 95% CI FID estimate. Requires Section 8 run 5A to succeed.

```
python scripts/eval_v2_multiseed.py --run-id 5a --num-samples 5000 --nfe 100
```

Evaluate run 5B (joint fine-tuned). Requires Section 8 run 5B to succeed.

```
python scripts/eval_v2_multiseed.py --run-id 5b --num-samples 5000 --nfe 100
```

Evaluate run 5C (path-KL ablation). Requires Section 8 run 5C to succeed.

```
python scripts/eval_v2_multiseed.py --run-id 5c --num-samples 5000 --nfe 100
```

### 11C — CIFAR-10 Baseline Samplers (DDPM, DDIM, SDE)

Evaluate DDPM, DDIM, and SDE samplers on CIFAR-10 at NFE={50,100,200,500} with N=10000 samples and eval seed 42. Requires Section 7 (CIFAR-10 score model) to succeed.

```
python scripts/run_baseline_evals.py --dataset cifar10 --model-ckpt checkpoints/score_cifar10_v2/score_best.pt --num-samples 10000 --seed 42 --nfe-list 50 100 200 500 --samplers ddpm ddim sde
```

### 11D — CIFAR-10 Controlled Sampler (Config D, 3-seed)

Evaluate the CIFAR-10 Config D controlled sampler across three seeds. Requires Section 9 (CIFAR-10 controlled training) to succeed.

```
python scripts/eval_controlled_multiseed.py --ctrl-ckpt checkpoints/cifar10_controlled_d_seed42/controlled_last.pt --score-ckpt checkpoints/score_cifar10_v2/score_best.pt --dataset cifar10 --num-samples 10000 --nfe 100
```

---

## Section 12 — Results and Report Generation

### 12A — LaTeX Results Table

Generate the LaTeX tabular and plain-text results table from all JSON result files in `data/results/`. Requires all evaluation steps in Section 11 to have written their result files.

```
python scripts/generate_results_table.py --results-dir data/results --output-dir data/results
```

### 12B — Paper Figures

Generate all publication-quality PNG and PDF figures (FID vs NFE, training curves, control energy scatter, work distributions, path KL trajectory). Requires Section 11 to succeed. Requires the previous command to succeed.

```
python scripts/generate_paper_figures.py --results-dir data/results --output-dir data/paper_figures
```

### 12C — Analytics HTML Report

Generate the self-contained HTML analytics report (inline SVG charts, sample grids, all metrics). Requires Section 12A and 12B to succeed.

```
python scripts/generate_report.py --output data/results/its_analytics_report.html
```

---

## Section 13 — Final Verification

Run the complete test suite one last time to confirm that all tests still pass after all training and evaluation artefacts have been written. This catches any file-path regressions introduced during the session.

```
python -m pytest tests/ -v
```
