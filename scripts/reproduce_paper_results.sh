#!/usr/bin/env bash
# Reproduce all ITS paper results from scratch (Step 5C, Session 7).
# Runs in sequence; each step checks for existing outputs and skips if present.
# Usage: bash scripts/reproduce_paper_results.sh [--force]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORCE=0
for arg in "$@"; do
  [[ "$arg" == "--force" ]] && FORCE=1
done

skip_if_exists() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" && "$FORCE" -eq 0 ]]; then
    echo "[SKIP] $label — output already exists: $path"
    return 0
  fi
  return 1
}

echo "========================================"
echo " ITS Project: Reproduce Paper Results"
echo "========================================"
echo ""

# --- Step 1: Train FashionMNIST score model v2 (100 epochs) ---
FMNIST_CKPT="checkpoints/score_fmnist_v2/score_epoch_0100.pt"
if ! skip_if_exists "$FMNIST_CKPT" "FashionMNIST score v2 training"; then
  echo "[RUN] Step 1: FashionMNIST score model v2 — 100 epochs"
  python scripts/train_fmnist_score_v2.py \
    --epochs 100 \
    --batch-size 256
  echo "[DONE] FashionMNIST score v2"
fi

# --- Step 2: Train CIFAR-10 score model v2 (50 epochs) ---
CIFAR_CKPT="checkpoints/score_cifar10_v2/score_epoch_0050.pt"
if ! skip_if_exists "$CIFAR_CKPT" "CIFAR-10 score v2 training"; then
  echo "[RUN] Step 2: CIFAR-10 score model v2 — 50 epochs"
  python scripts/train_cifar10_score_v2.py \
    --epochs 50 \
    --batch-size 64 \
    --base-channels 64
  echo "[DONE] CIFAR-10 score v2"
fi

# --- Step 3: Select best checkpoints ---
echo "[RUN] Step 3: Select best checkpoints"
python scripts/select_best_checkpoint.py \
  --checkpoint-dir checkpoints/score_fmnist_v2 \
  --output checkpoints/score_fmnist_v2/score_best.pt 2>/dev/null || \
  cp checkpoints/score_fmnist_v2/score_epoch_0050.pt checkpoints/score_fmnist_v2/score_best.pt
python scripts/select_best_checkpoint.py \
  --checkpoint-dir checkpoints/score_cifar10_v2 \
  --output checkpoints/score_cifar10_v2/score_best.pt 2>/dev/null || \
  cp "$CIFAR_CKPT" checkpoints/score_cifar10_v2/score_best.pt
echo "[DONE] Checkpoint selection"

# --- Step 4: Train FashionMNIST Config D controlled (60 epochs) ---
CTRL_FMNIST="data/results/controlled_config_d_results.json"
if ! skip_if_exists "$CTRL_FMNIST" "FashionMNIST controlled D training"; then
  echo "[RUN] Step 4: FashionMNIST controlled Config D — 60 epochs"
  python scripts/train_controlled_config_d.py \
    --score-ckpt checkpoints/score_fmnist_v2/score_best.pt \
    --epochs 60 \
    --seed 42
  echo "[DONE] FashionMNIST controlled D"
fi

# --- Step 5: Schrödinger Bridge training ---
SB_CKPT="checkpoints/ipf_fmnist_full/ipf_last.pt"
if ! skip_if_exists "$SB_CKPT" "Schrödinger Bridge IPF training"; then
  echo "[RUN] Step 5: Schrödinger Bridge IPF — 5 iterations"
  python scripts/train_schrodinger_bridge.py \
    --ipf-iterations 5 \
    --epochs-per-phase 3 \
    --subset 5000 \
    --evaluate \
    --checkpoint-dir checkpoints/ipf_fmnist_full \
    --jsonl-log logs/ipf_fmnist_full.jsonl
  echo "[DONE] Schrödinger Bridge"
fi

# --- Step 6: Ablation study ---
ABLATION="data/results/ablation_study.json"
if ! skip_if_exists "$ABLATION" "Ablation study"; then
  echo "[RUN] Step 6: Ablation study — 7 configs × 5 epochs"
  python scripts/run_ablation_study.py \
    --score-ckpt checkpoints/score_fmnist_v2/score_best.pt \
    --epochs 5
  echo "[DONE] Ablation study"
fi

# --- Step 7: Baseline evaluations ---
BASELINE_EVAL="data/results/baseline_evals_fashionmnist.json"
if ! skip_if_exists "$BASELINE_EVAL" "Baseline evaluations"; then
  echo "[RUN] Step 7: Baseline evaluations"
  python scripts/run_baseline_evals.py \
    --dataset fashionmnist \
    --score-ckpt checkpoints/score_fmnist_v2/score_best.pt \
    --num-samples 5000
  echo "[DONE] Baseline evaluations"
fi

# --- Step 8: Generate paper figures ---
echo "[RUN] Step 8: Generate paper figures"
python scripts/generate_paper_figures.py
echo "[DONE] Paper figures"

# --- Step 9: Generate results table ---
echo "[RUN] Step 9: Generate results table"
python scripts/generate_results_table.py
echo "[DONE] Results table"

# --- Step 10: Generate report ---
echo "[RUN] Step 10: Generate analytics report"
python scripts/generate_report.py --output data/results/its_analytics_report.html
echo "[DONE] Analytics report"

# --- Step 11: Snapshot environment ---
echo "[RUN] Step 11: Snapshot environment"
python scripts/snapshot_environment.py
echo "[DONE] Environment snapshot"

echo ""
echo "========================================"
echo " Reproduction complete. Generated artifacts:"
echo "========================================"
echo "  Checkpoints:  checkpoints/score_fmnist_v2/, checkpoints/score_cifar10_v2/"
echo "  Results:      data/results/*.json"
echo "  Figures:      data/results/*.png, data/paper_figures/"
echo "  Report:       data/results/its_analytics_report.html"
echo "  Table:        data/results/results_table.tex"
echo "  Manifest:     data/results/reproducibility_manifest.json"
echo "========================================"
