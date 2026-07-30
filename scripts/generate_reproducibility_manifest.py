"""Generate reproducibility manifest for all ITS paper results (Step 6E).

Scans data/results/ for all result JSON files and pairs them with:
  - the checkpoint that produced them
  - the seed used
  - the command to reproduce
  - environment details (git hash, PyTorch version, GPU)

Saves to data/results/reproducibility_manifest.json.

Usage:
    python scripts/generate_reproducibility_manifest.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _env_info() -> dict:
    env = {"git_commit": _git_hash()}
    try:
        import torch
        env["pytorch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    env["python_version"] = sys.version.split()[0]
    return env


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    results_dir = ROOT / "data" / "results"
    manifest: dict = {
        "environment": _env_info(),
        "results": [],
    }

    # Baseline evals
    for p in sorted(results_dir.glob("baseline_evals_*.json")):
        data = _load_json(p)
        if not data:
            continue
        manifest["results"].append({
            "file": str(p.relative_to(ROOT)),
            "description": f"Baseline evaluation — {data.get('dataset', '?')}",
            "checkpoint": data.get("checkpoint"),
            "num_samples": data.get("num_samples"),
            "seed": data.get("seed"),
            "command": (
                f"python scripts/run_baseline_evals.py "
                f"--dataset {data.get('dataset', 'fashionmnist')} "
                f"--model-ckpt {data.get('checkpoint', '<ckpt>')} "
                f"--num-samples {data.get('num_samples', 5000)} "
                f"--seed {data.get('seed', 42)}"
            ),
        })

    # Controlled Config D
    p = results_dir / "controlled_config_d_results.json"
    if p.is_file():
        data = _load_json(p)
        if data:
            seeds = [r.get("seed") for r in data.get("runs", [])]
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "Controlled training Config D — FashionMNIST",
                "seeds": seeds,
                "epochs": data.get("epochs"),
                "command": (
                    "python scripts/train_controlled_config_d.py "
                    "--score-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt "
                    "--epochs 30 --all-seeds"
                ),
            })

    # Ablation study
    p = results_dir / "ablation_study.json"
    if p.is_file():
        data = _load_json(p)
        if data:
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "Ablation study — 7 configs on FashionMNIST",
                "configs": [a.get("name") for a in data.get("ablations", [])],
                "epochs": data.get("epochs"),
                "num_samples": data.get("num_samples"),
                "command": (
                    "python scripts/run_ablation_study.py "
                    "--score-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt "
                    "--epochs 5"
                ),
            })

    # Crooks verification
    p = results_dir / "crooks_verification.json"
    if p.is_file():
        data = _load_json(p)
        if data:
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "Crooks fluctuation theorem verification",
                "passed": data.get("passed"),
                "command": "python scripts/run_crooks_verification.py --n-trajectories 5000",
            })

    # IPF toy convergence
    p = results_dir / "ipf_toy_convergence.json"
    if p.is_file():
        manifest["results"].append({
            "file": str(p.relative_to(ROOT)),
            "description": "IPF toy 2D Gaussian convergence test",
            "command": "python scripts/run_ipf_toy_convergence.py",
        })

    # Session 8: FashionMNIST eval matrix (3-seed, 2048 samples)
    p = results_dir / "fmnist_eval_matrix_final.json"
    if p.is_file():
        data = _load_json(p)
        if data:
            total = sum(len(v) for v in data.values() if isinstance(v, list))
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "FashionMNIST eval matrix — DDPM+DDIM × 4 NFE × 3 seeds, 2048 samples",
                "total_runs": total,
                "samplers": list(data.keys()),
                "command": (
                    "python scripts/run_fmnist_eval_matrix.py "
                    "--score-ckpt checkpoints/score_fmnist_v2/score_best.pt "
                    "--num-samples 2048"
                ),
            })

    # Session 8: Config D 3-seed CI
    p = results_dir / "controlled_config_d_multiseed.json"
    if p.is_file():
        data = _load_json(p)
        if data and "ci_95" in data:
            ci = data["ci_95"]
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "Config D controlled sampler 3-seed reliability CI",
                "fid_mean": ci.get("mean"),
                "fid_ci": [ci.get("lower"), ci.get("upper")],
                "epochs": data.get("epochs"),
                "command": (
                    "python scripts/eval_controlled_multiseed.py "
                    "--ctrl-ckpt checkpoints/controlled_config_d_seed42/controlled_last.pt "
                    "--score-ckpt checkpoints/score_fmnist_v2/score_best.pt"
                ),
            })

    # Session 8: Memorization check
    p = results_dir / "memorization_check.json"
    if p.is_file():
        data = _load_json(p)
        if data:
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "Memorization check — NN distance in pixel space",
                "fraction_below_threshold": data.get("fraction_below_threshold"),
                "memorization_concern": data.get("memorization_concern"),
                "command": (
                    "python scripts/compute_memorization_check.py "
                    "--score-ckpt checkpoints/score_fmnist_v2/score_best.pt"
                ),
            })

    # Session 9: ConvControl results (when available)
    p = results_dir / "controlled_conv_results.json"
    if p.is_file():
        data = _load_json(p)
        if data and "runs" in data:
            fid_vals = [r.get("fid") for r in data["runs"] if r.get("fid") is not None]
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "ConvControl 50-epoch controlled training — FashionMNIST",
                "fid_mean": sum(fid_vals) / len(fid_vals) if fid_vals else None,
                "epochs": data.get("epochs"),
                "command": (
                    "python scripts/train_controlled_config_d.py "
                    "--score-ckpt checkpoints/score_fmnist_v2/score_best.pt "
                    "--epochs 50 --use-conv-control --seed 42"
                ),
            })

    # Session 9: ConvControl 3-seed CI (when available)
    p = results_dir / "controlled_conv_multiseed.json"
    if p.is_file():
        data = _load_json(p)
        if data and "ci_95" in data:
            ci = data["ci_95"]
            manifest["results"].append({
                "file": str(p.relative_to(ROOT)),
                "description": "ConvControl 3-seed reliability CI",
                "fid_mean": ci.get("mean"),
                "fid_ci": [ci.get("lower"), ci.get("upper")],
                "command": (
                    "python scripts/eval_conv_multiseed.py "
                    "--ctrl-ckpt checkpoints/controlled_conv_seed42/controlled_last.pt "
                    "--score-ckpt checkpoints/score_fmnist_v2/score_best.pt"
                ),
            })

    out = results_dir / "reproducibility_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Reproducibility manifest: {out}")
    print(f"  Environment: {manifest['environment']}")
    print(f"  Result entries: {len(manifest['results'])}")


if __name__ == "__main__":
    main()
