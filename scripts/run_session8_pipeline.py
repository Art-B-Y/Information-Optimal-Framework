"""Session 8 post-eval pipeline — runs after fmnist_eval_matrix_final.json is complete.

Steps:
1. Controlled Config D multi-seed evaluation (3 seeds, 2048 samples)
2. Entropy production profile
3. Memorization check
4. Mode coverage
5. Regenerate paper figures
6. Regenerate results table
7. Regenerate analytics report

Usage:
    python scripts/run_session8_pipeline.py
    python scripts/run_session8_pipeline.py --skip-analyses  # skip entropy/memorization/mode
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_step(label: str, cmd: list[str], skip_if: Path | None = None) -> bool:
    if skip_if and skip_if.exists():
        print(f"[SKIP] {label} — output exists: {skip_if.name}")
        return True
    print(f"\n{'='*60}")
    print(f"[STEP] {label}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAILED] {label} (exit {result.returncode})")
        return False
    print(f"[DONE] {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-analyses", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    py = sys.executable

    score_ckpt = ROOT / "checkpoints" / "score_fmnist_v2" / "score_best.pt"
    ctrl_ckpt = ROOT / "checkpoints" / "controlled_config_d_seed42" / "controlled_last.pt"

    if not score_ckpt.exists():
        print(f"ERROR: score checkpoint not found: {score_ckpt}")
        sys.exit(1)
    if not ctrl_ckpt.exists():
        print(f"ERROR: ctrl checkpoint not found: {ctrl_ckpt}")
        sys.exit(1)

    # Step 1: Controlled Config D multi-seed eval
    run_step(
        "Controlled Config D multi-seed (3 seeds, 2048 samples)",
        [py, "scripts/eval_controlled_multiseed.py",
         "--ctrl-ckpt", str(ctrl_ckpt),
         "--score-ckpt", str(score_ckpt),
         "--num-samples", "2048",
         "--nfe", "100"],
    )

    if not args.skip_analyses:
        # Step 2: Entropy production profile
        run_step(
            "Entropy production profile",
            [py, "scripts/compute_entropy_production.py",
             "--score-ckpt", str(score_ckpt),
             "--ctrl-ckpt", str(ctrl_ckpt),
             "--n-trajectories", "256",
             "--nfe", "100"],
            skip_if=ROOT / "data" / "results" / "entropy_production_profile.json",
        )

        # Step 3: Memorization check
        run_step(
            "Memorization check (1000 samples)",
            [py, "scripts/compute_memorization_check.py",
             "--score-ckpt", str(score_ckpt),
             "--num-samples", "1000"],
            skip_if=ROOT / "data" / "results" / "memorization_check.json",
        )

        # Step 4: Mode coverage
        run_step(
            "Mode coverage analysis",
            [py, "scripts/compute_mode_coverage.py",
             "--score-ckpt", str(score_ckpt),
             "--num-samples", "1000"],
            skip_if=ROOT / "data" / "results" / "mode_coverage.json",
        )

    # Step 5: Regenerate paper figures
    run_step(
        "Generate paper figures",
        [py, "scripts/generate_paper_figures.py"],
    )

    # Step 6: Regenerate results table
    run_step(
        "Generate results table",
        [py, "scripts/generate_results_table.py"],
    )

    # Step 7: Regenerate analytics report
    run_step(
        "Generate analytics report",
        [py, "scripts/generate_report.py",
         "--output", "data/results/its_analytics_report.html"],
    )

    # Step 8: Environment snapshot
    run_step(
        "Environment snapshot",
        [py, "scripts/snapshot_environment.py"],
    )

    print("\n" + "="*60)
    print("Session 8 pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()
