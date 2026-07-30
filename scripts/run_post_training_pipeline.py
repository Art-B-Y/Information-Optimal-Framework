"""Post-training pipeline — runs after FashionMNIST v2 epoch 50 is available.

Runs in order:
  1. Baseline evals (DDPM/DDIM/SDE, NFE={50,100,200,500})
  2. Controlled Config D (3 seeds, 10 epochs for speed)
  3. Ablation study (6 configs, 3 epochs each)
  4. Regenerate report, paper figures, results table
  5. Launch CIFAR-10 v2 training (background)

Usage:
    python scripts/run_post_training_pipeline.py
    python scripts/run_post_training_pipeline.py --quick   # smoke test (N=256, 1 epoch)
    python scripts/run_post_training_pipeline.py --score-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"WARNING: {label} exited with code {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt",
                        default="checkpoints/score_fmnist_v2/score_epoch_0050.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: N=256, 1 epoch ablation, 2 seeds")
    args = parser.parse_args()

    ckpt = args.score_ckpt
    if not Path(ROOT / ckpt).is_file():
        print(f"ERROR: Checkpoint not found: {ckpt}")
        print("Run train_fmnist_score_v2.py first and wait for epoch 50.")
        sys.exit(1)

    py = sys.executable
    dev_args = ["--device", args.device] if args.device else []
    n_samples = "256" if args.quick else "5000"
    n_epochs_d = "2" if args.quick else "10"
    n_epochs_abl = "1" if args.quick else "5"
    seeds = ["--seed", "42"] if args.quick else ["--all-seeds"]

    # 1. Baseline evals
    run([py, "scripts/run_baseline_evals.py",
         "--dataset", "fashionmnist",
         "--model-ckpt", ckpt,
         "--num-samples", n_samples,
         "--nfe-list", "50", "100", "200", "500",
         "--seed", "42",
         ] + dev_args,
        "Baseline evals: DDPM/DDIM/SDE at NFE={50,100,200,500}")

    # 2. Config D controlled training (3 seeds)
    run([py, "scripts/train_controlled_config_d.py",
         "--score-ckpt", ckpt,
         "--epochs", n_epochs_d,
         ] + seeds + dev_args,
        f"Controlled Config D training ({n_epochs_d} epochs)")

    # 3. Ablation study
    run([py, "scripts/run_ablation_study.py",
         "--score-ckpt", ckpt,
         "--epochs", n_epochs_abl,
         ] + dev_args,
        f"Ablation study (6 configs × {n_epochs_abl} epochs)")

    # 4. Control drift analysis (from Config D JSONL)
    config_d_jsonl = ROOT / "logs" / "controlled_config_d_seed42.jsonl"
    if config_d_jsonl.is_file():
        run([py, "scripts/run_control_drift_analysis.py",
             "--from-jsonl", str(config_d_jsonl)],
            "Control drift analysis from Config D JSONL")
    else:
        print("Skipping control drift analysis — controlled_config_d_seed42.jsonl not found")

    # 5. IPF FashionMNIST 5 iterations (Step 4B)
    if not args.quick:
        run([py, "scripts/train_schrodinger_bridge.py",
             "--ipf-iterations", "5",
             "--epochs-per-phase", "3",
             "--subset", "5000",
             "--evaluate",
             "--checkpoint-dir", "checkpoints/ipf_fmnist_full",
             "--jsonl-log", "logs/ipf_fmnist_full.jsonl",
             ] + dev_args,
            "IPF FashionMNIST 5 iterations (Step 4B)")

    # 6. Regenerate analytics artifacts
    run([py, "scripts/generate_report.py",
         "--output", "data/results/its_analytics_report.html"],
        "Regenerate HTML analytics report")

    run([py, "scripts/generate_paper_figures.py"],
        "Regenerate paper figures")

    run([py, "scripts/generate_results_table.py"],
        "Regenerate LaTeX results table")

    run([py, "scripts/generate_reproducibility_manifest.py"],
        "Regenerate reproducibility manifest")

    # 7. Launch CIFAR-10 v2 training in background
    print(f"\n{'='*60}")
    print("STEP: Launch CIFAR-10 v2 training (background, ~3-4hrs)")
    print("To launch CIFAR-10 training, run:")
    print("  python scripts/train_cifar10_score_v2.py --epochs 50 --batch-size 64"
          " > logs/cifar10_score_v2.log 2>&1")

    print("\n" + "="*60)
    print("Post-training pipeline complete!")
    print("Results in data/results/:")
    print("  baseline_evals_fashionmnist.json")
    print("  controlled_config_d_results.json")
    print("  ablation_study.json")
    print("  control_drift_analysis.json")
    print("  pareto_frontier_final.json")
    print("  reproducibility_manifest.json")
    print("  its_analytics_report.html")
    print("  results_table.tex")
    print("Paper figures in data/paper_figures/")


if __name__ == "__main__":
    main()
