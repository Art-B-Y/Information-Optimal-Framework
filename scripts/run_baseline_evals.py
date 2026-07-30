"""Baseline evaluations: DDPM/DDIM/SDE at multiple NFE settings.

Step 2C (Session 6). Evaluates a trained score model with three samplers
(DDPM, DDIM, SDE) at NFE = {50, 100, 200, 500}, N=5000, seed=42.

Saves results to data/results/baseline_evals_{dataset}.json.

Usage:
    python scripts/run_baseline_evals.py --dataset fashionmnist --model-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt
    python scripts/run_baseline_evals.py --dataset cifar10 --model-ckpt checkpoints/score_cifar10_v2/score_epoch_0050.pt
    python scripts/run_baseline_evals.py --dataset fashionmnist --model-ckpt <ckpt> --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashionmnist", choices=["fashionmnist", "cifar10"])
    parser.add_argument("--model-ckpt", required=True, help="Path to trained score model checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nfe-list", nargs="+", type=int, default=[50, 100, 200, 500],
                        help="NFE values to evaluate at")
    parser.add_argument("--samplers", nargs="+", default=["ddpm", "ddim", "sde"],
                        choices=["ddpm", "ddim", "sde"])
    parser.add_argument("--quick", action="store_true", help="Use N=256 for smoke testing")
    args = parser.parse_args()

    import torch
    from its.models import ScoreUNetConfig, build_score_model
    from its.eval import EvaluationConfig
    from its.eval.evaluator import evaluate_sampler, evaluate_ddpm_baseline
    from its.sde import ScoreSDEConfig

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    num_samples = 256 if args.quick else args.num_samples

    print(f"Baseline evals: dataset={args.dataset}, device={device}, N={num_samples}")
    print(f"Samplers: {args.samplers}, NFE: {args.nfe_list}")
    print(f"Checkpoint: {args.model_ckpt}")

    # Load checkpoint and build model
    ckpt_path = Path(args.model_ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Detect model config from checkpoint
    model_cfg_dict = state.get("model_config") or state.get("score_config")
    if model_cfg_dict is not None:
        model_cfg = ScoreUNetConfig(**model_cfg_dict)
    else:
        # Fallback: infer from dataset
        if args.dataset == "fashionmnist":
            model_cfg = ScoreUNetConfig(
                in_channels=1,
                base_channels=64,
                channel_mults=(1, 2, 2),
                use_time_embedding=True,
                time_embed_dim=128,
            )
        else:
            model_cfg = ScoreUNetConfig(
                in_channels=3,
                base_channels=64,
                channel_mults=(1, 2, 2, 4),
                use_time_embedding=True,
                time_embed_dim=128,
                use_attention=True,
                attention_heads=4,
            )

    score_model = build_score_model(model_cfg).to(device)

    # Load weights: prefer EMA shadow weights (better quality), fall back to raw model state dict.
    # ExponentialMovingAverage.state_dict() returns {"decay": float, "shadow": [tensor, ...]}.
    # The "shadow" list matches model.parameters() in order.
    ema_sd = state.get("ema_state_dict")
    if ema_sd and "shadow" in ema_sd:
        # Apply EMA shadow weights by iterating over model parameters
        shadow_tensors = ema_sd["shadow"]
        trainable_params = [p for p in score_model.parameters() if p.requires_grad]
        if len(shadow_tensors) == len(trainable_params):
            with torch.no_grad():
                for param, shadow in zip(trainable_params, shadow_tensors):
                    param.data.copy_(shadow.to(device))
            print("Loaded EMA shadow weights.")
        else:
            print(f"WARNING: EMA shadow length mismatch ({len(shadow_tensors)} vs {len(trainable_params)}), "
                  "falling back to model_state_dict")
            ema_sd = None
    if not ema_sd or "shadow" not in ema_sd:
        score_sd = (
            state.get("score_state_dict")
            or state.get("model_state_dict")
            or state
        )
        score_model.load_state_dict(score_sd, strict=True)
        print("Loaded raw model state dict.")
    score_model.eval()

    results: dict = {
        "dataset": args.dataset,
        "checkpoint": str(args.model_ckpt),
        "num_samples": num_samples,
        "seed": args.seed,
        "runs": [],
    }

    for nfe in args.nfe_list:
        sde_cfg = ScoreSDEConfig(
            beta_min=0.1,
            beta_max=10.0,
            num_steps=nfe,
            sigma_min=0.01,
            sigma_max=1.0,
        )
        eval_cfg = EvaluationConfig(
            dataset_name=args.dataset,
            num_samples=num_samples,
            batch_size=args.batch_size,
            device=device_str,
            seed=args.seed,
        )

        for sampler_name in args.samplers:
            print(f"  Evaluating {sampler_name} at NFE={nfe}...", end=" ", flush=True)
            try:
                if sampler_name == "sde":
                    metrics = evaluate_sampler(score_model, None, sde_cfg, eval_cfg)
                else:
                    metrics = evaluate_ddpm_baseline(score_model, sde_cfg, eval_cfg, baseline=sampler_name)

                rec = {
                    "sampler": sampler_name,
                    "nfe": nfe,
                    **metrics,
                }
                results["runs"].append(rec)
                print(f"FID={metrics['fid']:.2f} IS={metrics['inception_score_mean']:.3f}")
            except Exception as e:
                print(f"FAILED: {e}")
                results["runs"].append({"sampler": sampler_name, "nfe": nfe, "error": str(e)})

    out_path = ROOT / "data" / "results" / f"baseline_evals_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out_path}")

    # Print summary table
    print(f"\n{'Sampler':<8} {'NFE':<6} {'FID':>8} {'IS':>8}")
    print("-" * 32)
    for r in results["runs"]:
        if "error" not in r:
            print(f"{r['sampler']:<8} {r['nfe']:<6} {r['fid']:>8.2f} {r['inception_score_mean']:>8.3f}")


if __name__ == "__main__":
    main()
