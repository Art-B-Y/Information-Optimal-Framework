"""Generate the definitive FashionMNIST evaluation matrix (Step 3A, Session 7).

Evaluates all samplers at all NFE levels with 3 seeds. Adds 95% CI.
Writes fmnist_eval_matrix_final.json.

Resumable: skips runs already completed with sufficient samples.
Saves after each individual run to protect against interruption.

Usage:
    python scripts/run_fmnist_eval_matrix.py --score-ckpt checkpoints/score_fmnist_v2/score_best.pt
    python scripts/run_fmnist_eval_matrix.py --score-ckpt <ckpt> --num-samples 5000 --quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEEDS = [42, 123, 7]
NFE_LEVELS = [50, 100, 200, 500]
OUT_PATH = ROOT / "data" / "results" / "fmnist_eval_matrix_final.json"


def ci_95(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"lower": float("nan"), "upper": float("nan"), "mean": float("nan"), "std": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"lower": mean, "upper": mean, "mean": mean, "std": 0.0}
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    half_width = 1.96 * std / math.sqrt(n)
    return {"lower": mean - half_width, "upper": mean + half_width,
            "mean": mean, "std": std}


def load_existing(min_samples: int) -> dict:
    """Load existing matrix, filtering out entries with insufficient samples."""
    if not OUT_PATH.exists():
        return {"ddpm": [], "ddim": []}
    try:
        data = json.loads(OUT_PATH.read_text())
        result = {"ddpm": [], "ddim": []}
        for sampler in ("ddpm", "ddim"):
            for entry in data.get(sampler, []):
                if entry.get("num_samples", 0) >= min_samples:
                    result[sampler].append(entry)
        return result
    except Exception:
        return {"ddpm": [], "ddim": []}


def save_matrix(matrix: dict) -> None:
    OUT_PATH.write_text(json.dumps(matrix, indent=2))


def already_done(matrix: dict, sampler: str, nfe: int, seed: int) -> bool:
    return any(
        e.get("nfe") == nfe and e.get("seed") == seed
        for e in matrix.get(sampler, [])
    )


def recompute_ci(matrix: dict, sampler: str, nfe: int) -> None:
    entries = [e for e in matrix[sampler] if e["nfe"] == nfe]
    fids = [e["fid"] for e in entries]
    if len(fids) >= 2:
        ci = ci_95(fids)
        for e in entries:
            e["ci_95"] = ci


def eval_ddpm_one(score_model, sde_cfg, nfe: int, num_samples: int, seed: int,
                  device: torch.device) -> dict:
    from its.eval.evaluator import evaluate_ddpm_baseline, EvaluationConfig
    from its.sde import ScoreSDEConfig

    nfe_cfg = ScoreSDEConfig(
        beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max,
        num_steps=nfe, sigma_min=sde_cfg.sigma_min, sigma_max=sde_cfg.sigma_max,
    )
    eval_cfg = EvaluationConfig(dataset_name="fashionmnist", num_samples=num_samples,
                                batch_size=64, device=str(device), seed=seed)
    result = evaluate_ddpm_baseline(score_model, nfe_cfg, eval_cfg, baseline="ddpm")
    return {"fid": result["fid"], "nfe": nfe, "num_samples": num_samples, "seed": seed}


def eval_ddim_one(score_model, sde_cfg, nfe: int, num_samples: int, seed: int,
                  device: torch.device) -> dict:
    from its.eval.evaluator import evaluate_ddpm_baseline, EvaluationConfig
    from its.sde import ScoreSDEConfig

    nfe_cfg = ScoreSDEConfig(
        beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max,
        num_steps=nfe, sigma_min=sde_cfg.sigma_min, sigma_max=sde_cfg.sigma_max,
    )
    eval_cfg = EvaluationConfig(dataset_name="fashionmnist", num_samples=num_samples,
                                batch_size=64, device=str(device), seed=seed)
    result = evaluate_ddpm_baseline(score_model, nfe_cfg, eval_cfg, baseline="ddim")
    return {"fid": result["fid"], "nfe": nfe, "num_samples": num_samples, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--quick", action="store_true",
                        help="Use 256 samples and 1 seed for smoke test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.quick:
        args.num_samples = 256
        seeds = [42]
        nfe_levels = [50, 200]
    else:
        seeds = SEEDS
        nfe_levels = NFE_LEVELS

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from its.models import ScoreUNetConfig, build_score_model
    from its.sde import ScoreSDEConfig

    state = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                 use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=100,
                              sigma_min=0.01, sigma_max=1.0)

    # Load existing results (resume support)
    matrix = load_existing(args.num_samples)
    existing_ddpm = sum(1 for e in matrix["ddpm"] if e.get("num_samples", 0) >= args.num_samples)
    existing_ddim = sum(1 for e in matrix["ddim"] if e.get("num_samples", 0) >= args.num_samples)
    total_needed = len(nfe_levels) * len(seeds) * 2
    existing_total = existing_ddpm + existing_ddim
    if existing_total > 0:
        print(f"Resuming: {existing_total}/{total_needed} runs already complete at {args.num_samples} samples.")

    # DDPM evaluation
    for nfe in nfe_levels:
        for seed in seeds:
            if already_done(matrix, "ddpm", nfe, seed):
                print(f"SKIP DDPM NFE={nfe} seed={seed} (already done)")
                continue
            print(f"DDPM NFE={nfe} seed={seed}...", end=" ", flush=True)
            try:
                entry = eval_ddpm_one(model, sde_cfg, nfe, args.num_samples, seed, device)
                print(f"FID={entry['fid']:.2f}")
                matrix["ddpm"].append({**entry, "ci_95": ci_95([entry["fid"]])})
                recompute_ci(matrix, "ddpm", nfe)
                save_matrix(matrix)
            except Exception as e:
                print(f"ERROR: {e}")

    # DDIM evaluation
    for nfe in nfe_levels:
        for seed in seeds:
            if already_done(matrix, "ddim", nfe, seed):
                print(f"SKIP DDIM NFE={nfe} seed={seed} (already done)")
                continue
            print(f"DDIM NFE={nfe} seed={seed}...", end=" ", flush=True)
            try:
                entry = eval_ddim_one(model, sde_cfg, nfe, args.num_samples, seed, device)
                print(f"FID={entry['fid']:.2f}")
                matrix["ddim"].append({**entry, "ci_95": ci_95([entry["fid"]])})
                recompute_ci(matrix, "ddim", nfe)
                save_matrix(matrix)
            except Exception as e:
                print(f"ERROR: {e}")

    # Final CI recompute for all NFE levels
    for sampler in ("ddpm", "ddim"):
        for nfe in nfe_levels:
            recompute_ci(matrix, sampler, nfe)
    save_matrix(matrix)

    total_done = sum(1 for e in matrix["ddpm"] if e.get("num_samples", 0) >= args.num_samples)
    total_done += sum(1 for e in matrix["ddim"] if e.get("num_samples", 0) >= args.num_samples)
    print(f"\nEvaluation matrix complete: {total_done}/{total_needed} runs at {args.num_samples} samples")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
