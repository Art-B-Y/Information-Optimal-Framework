"""Session 7 post-training pipeline — runs sequentially after FashionMNIST score v2.

Steps (in order):
1. Select best FashionMNIST score v2 checkpoint
2. Resume FashionMNIST Config D controlled training (30→60 epochs)
3. Train CIFAR-10 score model v2 (50 epochs)
4. Extend SB IPF training (5→10 iterations)
5. Run FashionMNIST eval matrix (quick 256-sample version for CI check)
6. Generate paper figures and report

Run after FashionMNIST score v2 training completes.

Usage:
    python scripts/run_session7_pipeline.py
    python scripts/run_session7_pipeline.py --skip-cifar10  # skip slow CIFAR-10 run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_step(label: str, cmd: list[str]) -> bool:
    print(f"\n{'='*60}")
    print(f"[STEP] {label}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAILED] {label} (exit code {result.returncode})")
        return False
    print(f"[DONE] {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cifar10", action="store_true")
    parser.add_argument("--skip-sb", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    py = sys.executable

    # Step 1: Select best FashionMNIST score v2 checkpoint
    fmnist_best = ROOT / "checkpoints" / "score_fmnist_v2" / "score_best.pt"
    if not fmnist_best.exists():
        # Try to select via JSONL log; fall back to latest epoch
        jsonl = ROOT / "logs" / "fmnist_score_v2.jsonl"
        ckpt_dir = ROOT / "checkpoints" / "score_fmnist_v2"
        if jsonl.exists():
            ok = run_step("Select best FashionMNIST score checkpoint", [
                py, "scripts/select_best_checkpoint.py",
                "--jsonl", str(jsonl),
                "--checkpoint-dir", str(ckpt_dir),
                "--output", str(fmnist_best),
            ])
        if not fmnist_best.exists():
            # Fall back: pick latest
            candidates = sorted(ckpt_dir.glob("score_epoch_*.pt"))
            if candidates:
                import shutil
                shutil.copy2(candidates[-1], fmnist_best)
                print(f"  Fallback: copied {candidates[-1].name} -> score_best.pt")

    if not fmnist_best.exists():
        print("ERROR: No FashionMNIST score checkpoint found. Train first.")
        sys.exit(1)

    print(f"\nUsing score checkpoint: {fmnist_best}")

    # Step 2: Resume FashionMNIST Config D controlled (to 60 epochs)
    ctrl_ckpt_dir = ROOT / "checkpoints" / "controlled_config_d_seed42"
    ctrl_done = ROOT / "data" / "results" / "controlled_config_d_session7.json"
    if not ctrl_done.exists():
        cmd = [
            py, "scripts/train_controlled_config_d.py",
            "--score-ckpt", str(fmnist_best),
            "--epochs", "60",
            "--seed", "42",
            "--num-samples-eval", "2048",
        ]
        # Add device if specified
        if args.device:
            cmd += ["--device", args.device]
        ok = run_step("FashionMNIST Config D — resume to 60 epochs", cmd)
        # Rename output to session7 version
        orig = ROOT / "data" / "results" / "controlled_config_d_results.json"
        if orig.exists():
            import shutil
            shutil.copy2(orig, ctrl_done)

    # Step 3: CIFAR-10 score model
    if not args.skip_cifar10:
        cifar_ckpt = ROOT / "checkpoints" / "score_cifar10_v2" / "score_epoch_0050.pt"
        if not cifar_ckpt.exists():
            cmd = [
                py, "scripts/train_cifar10_score_v2.py",
                "--epochs", "50",
                "--batch-size", "64",
                "--base-channels", "64",
            ]
            if args.device:
                cmd += ["--device", args.device]
            ok = run_step("CIFAR-10 score model v2 — 50 epochs", cmd)

    # Step 4: Extend SB IPF (5 → 10 iterations)
    if not args.skip_sb:
        sb_iter10 = ROOT / "checkpoints" / "ipf_fmnist_full" / "ipf_iter_0010.pt"
        if not sb_iter10.exists():
            cmd = [
                py, "scripts/train_schrodinger_bridge.py",
                "--ipf-iterations", "5",
                "--epochs-per-phase", "3",
                "--subset", "5000",
                "--checkpoint-dir", str(ROOT / "checkpoints" / "ipf_fmnist_full"),
                "--jsonl-log", str(ROOT / "logs" / "ipf_fmnist_full.jsonl"),
                "--resume-from", str(ROOT / "checkpoints" / "ipf_fmnist_full" / "ipf_last.pt"),
            ]
            run_step("Extend SB IPF (5 more iterations)", cmd)

    # Step 5: Run eval matrix (quick version for now)
    matrix_file = ROOT / "data" / "results" / "fmnist_eval_matrix_final.json"
    if not matrix_file.exists():
        cmd = [
            py, "scripts/run_fmnist_eval_matrix.py",
            "--score-ckpt", str(fmnist_best),
            "--quick",
        ]
        run_step("FashionMNIST eval matrix (quick)", cmd)

    # Step 6: Regenerate figures and report
    run_step("Generate paper figures", [py, "scripts/generate_paper_figures.py"])
    run_step("Generate results table", [py, "scripts/generate_results_table.py"])
    run_step("Generate report", [py, "scripts/generate_report.py",
                                  "--output", "data/results/its_analytics_report.html"])
    run_step("Snapshot environment", [py, "scripts/snapshot_environment.py"])

    print("\n" + "="*60)
    print("Session 7 pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()
