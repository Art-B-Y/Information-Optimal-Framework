"""Run ConvControl sampler evaluation across 3 seeds for 95% CI (Session 9).

Uses the best ConvControl checkpoint (controlled_conv_seed42) evaluated at
seeds 42, 123, 7 to produce a 95% CI for the controlled sampler FID.

Usage:
    python scripts/eval_conv_multiseed.py \
        --ctrl-ckpt checkpoints/controlled_conv_seed42/controlled_last.pt \
        --score-ckpt checkpoints/score_fmnist_v2/score_best.pt
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
OUT_PATH = ROOT / "data" / "results" / "controlled_conv_multiseed.json"


def ci_95(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"lower": mean, "upper": mean, "mean": mean, "std": 0.0}
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    half_width = 1.96 * std / math.sqrt(n)
    return {"lower": mean - half_width, "upper": mean + half_width, "mean": mean, "std": std}


def eval_one(score_model, ctrl_policy, sde_cfg, num_samples: int, seed: int,
             nfe: int, device: torch.device) -> dict:
    from its.eval.evaluator import evaluate_sampler, EvaluationConfig
    from its.sde import ScoreSDEConfig

    nfe_cfg = ScoreSDEConfig(
        beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max,
        num_steps=nfe, sigma_min=sde_cfg.sigma_min, sigma_max=sde_cfg.sigma_max,
    )
    eval_cfg = EvaluationConfig(
        dataset_name="fashionmnist", num_samples=num_samples,
        batch_size=64, device=str(device), seed=seed,
    )
    result = evaluate_sampler(score_model, ctrl_policy, nfe_cfg, eval_cfg)
    return {
        "fid": result["fid"],
        "inception_score_mean": result.get("inception_score_mean"),
        "nfe": nfe, "num_samples": num_samples, "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl-ckpt", required=True)
    parser.add_argument("--score-ckpt", required=True)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs of the ConvControl checkpoint")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy
    from its.sde import ScoreSDEConfig

    score_state = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                use_time_embedding=True, time_embed_dim=128)
    score_model = build_score_model(model_cfg).to(device)
    sd = score_state.get("model_state_dict") or score_state.get("score_state_dict")
    score_model.load_state_dict(sd, strict=True)
    score_model.eval()

    ctrl_state = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
    ctrl_cfg_dict = ctrl_state.get("control_config", {})
    conv_cfg = ConvControlConfig(
        in_channels=ctrl_cfg_dict.get("in_channels", 1),
        base_channels=ctrl_cfg_dict.get("base_channels", 64),
        num_blocks=ctrl_cfg_dict.get("num_blocks", 3),
        time_embed_dim=ctrl_cfg_dict.get("time_embed_dim", 128),
        dropout=ctrl_cfg_dict.get("dropout", 0.0),
    )
    ctrl_policy = build_conv_control_policy(conv_cfg, device=device)
    ctrl_policy.load_state_dict(ctrl_state["control_state_dict"], strict=True)
    ctrl_policy.eval()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=args.nfe,
                              sigma_min=0.01, sigma_max=1.0)

    # Load existing results to allow resume
    results: list[dict] = []
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
            results = existing.get("runs", [])
            done_seeds = {r["seed"] for r in results if r.get("num_samples", 0) >= args.num_samples}
            print(f"Resuming: {len(done_seeds)}/3 seeds already complete.")
        except Exception:
            results = []

    done_seeds = {r["seed"] for r in results if r.get("num_samples", 0) >= args.num_samples}

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"SKIP seed={seed} (already done)")
            continue
        print(f"Evaluating seed={seed} NFE={args.nfe} N={args.num_samples}...", end=" ", flush=True)
        try:
            entry = eval_one(score_model, ctrl_policy, sde_cfg, args.num_samples, seed, args.nfe, device)
            print(f"FID={entry['fid']:.2f}")
            results.append(entry)
            fids = [r["fid"] for r in results if r.get("num_samples", 0) >= args.num_samples]
            OUT_PATH.write_text(json.dumps({
                "config": "D_conv", "epochs": args.epochs, "nfe": args.nfe,
                "num_samples": args.num_samples, "runs": results,
                "ci_95": ci_95(fids),
            }, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")

    fids = [r["fid"] for r in results if r.get("num_samples", 0) >= args.num_samples]
    if fids:
        ci = ci_95(fids)
        print(f"\nConvControl FID (N={args.num_samples}, {len(fids)} seeds): {ci['mean']:.2f} ± {ci['std']:.2f}")
        print(f"  95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
        print(f"  Seeds: {[r['fid'] for r in results]}")
        print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
