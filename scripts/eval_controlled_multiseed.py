"""Run Config D controlled sampler evaluation across 3 seeds.

Reads model architecture from the checkpoint's saved config keys so it works
for both FashionMNIST (in_channels=1) and CIFAR-10 (in_channels=3) runs.

Usage:
    # FashionMNIST (Session 8 Config D):
    python scripts/eval_controlled_multiseed.py \
        --ctrl-ckpt checkpoints/controlled_config_d_seed42/controlled_last.pt \
        --score-ckpt checkpoints/score_fmnist_v2/score_best.pt \
        --dataset fashionmnist

    # CIFAR-10 (Session 9 Config D):
    python scripts/eval_controlled_multiseed.py \
        --ctrl-ckpt checkpoints/cifar10_controlled_d_seed42/controlled_last.pt \
        --score-ckpt checkpoints/score_cifar10_v2/score_best.pt \
        --dataset cifar10 --num-samples 10000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEEDS = [42, 123, 7]


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
             nfe: int, device: torch.device, dataset: str) -> dict:
    from its.eval.evaluator import evaluate_sampler, EvaluationConfig
    from its.sde import ScoreSDEConfig

    # Audit 2026-07-15 (A3): this rebuilt the config copying five fields and
    # DROPPING control_weight, silently resetting it to its 0.0 default -- which
    # disabled the control branch entirely, so every "controlled" FID in the
    # pre-audit record is really the uncontrolled sampler's.  Use dataclasses
    # .replace so no field can be dropped by omission again.
    nfe_cfg = dataclasses.replace(sde_cfg, num_steps=nfe)
    assert nfe_cfg.control_weight > 0, (
        "control_weight must be > 0 for a controlled evaluation; "
        f"got {nfe_cfg.control_weight}."
    )
    eval_cfg = EvaluationConfig(
        dataset_name=dataset, num_samples=num_samples,
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", default="fashionmnist",
                        choices=["fashionmnist", "cifar10"],
                        help="Dataset to use for FID computation (default: fashionmnist)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy
    from its.sde import ScoreSDEConfig

    # --- Score model ---
    score_state = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    score_cfg_dict = score_state.get("score_config") or score_state.get("model_config")
    if score_cfg_dict:
        model_cfg = ScoreUNetConfig(**score_cfg_dict)
    elif args.dataset == "cifar10":
        model_cfg = ScoreUNetConfig(
            in_channels=3, base_channels=64, channel_mults=(1, 2, 2, 4),
            use_time_embedding=True, time_embed_dim=128,
            use_attention=True, attention_heads=4,
        )
    else:
        model_cfg = ScoreUNetConfig(
            in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
            use_time_embedding=True, time_embed_dim=128,
        )
    score_model = build_score_model(model_cfg).to(device)
    score_sd = (score_state.get("score_state_dict") or
                score_state.get("model_state_dict") or score_state)
    score_model.load_state_dict(score_sd, strict=True)
    score_model.eval()

    # --- Control policy ---
    ctrl_state = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
    ctrl_cfg_dict = ctrl_state.get("control_config", {})
    if ctrl_cfg_dict:
        ctrl_cfg = ControlConfig(**ctrl_cfg_dict)
    elif args.dataset == "cifar10":
        ctrl_cfg = ControlConfig(state_dim=3072, hidden_dim=512, depth=4, time_embedding_dim=16)
    else:
        ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=256, depth=3, time_embedding_dim=16)

    image_shape = (3, 32, 32) if args.dataset == "cifar10" else (1, 28, 28)
    ctrl_policy = build_control_policy(ctrl_cfg, device=device, image_shape=image_shape)
    ctrl_policy.load_state_dict(ctrl_state["control_state_dict"], strict=True)
    ctrl_policy.eval()

    ckpt_epoch = ctrl_state.get("epoch", "?")
    print(f"Loaded ctrl checkpoint (epoch={ckpt_epoch})")

    # Audit 2026-07-15: control_weight=1.0 must be explicit or the control is a
    # silent no-op (A3).  beta_max was 5.0 here while training used 10.0 (C6);
    # under the corrected VE parameterisation beta is unused for sampling, but
    # the value is aligned to avoid resurrecting the mismatch.
    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=args.nfe,
                              sigma_min=0.01, sigma_max=1.0,
                              control_weight=1.0)

    out_path = ROOT / "data" / "results" / f"controlled_config_d_{args.dataset}_multiseed.json"

    # Load existing results to allow resume.
    results: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            results = existing.get("runs", [])
            print(f"Resuming: {len(results)}/3 seeds already complete.")
        except Exception:
            results = []

    done_seeds = {r["seed"] for r in results if r.get("num_samples", 0) >= args.num_samples}

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"SKIP seed={seed} (already done)")
            continue
        print(f"Evaluating seed={seed} NFE={args.nfe} N={args.num_samples}...", end=" ", flush=True)
        try:
            entry = eval_one(score_model, ctrl_policy, sde_cfg, args.num_samples,
                             seed, args.nfe, device, args.dataset)
            print(f"FID={entry['fid']:.2f}")
            results.append(entry)
            fids = [r["fid"] for r in results if r.get("num_samples", 0) >= args.num_samples]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "config": "D", "dataset": args.dataset, "nfe": args.nfe,
                "num_samples": args.num_samples, "runs": results,
                "ci_95": ci_95(fids),
            }, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")

    fids = [r["fid"] for r in results if r.get("num_samples", 0) >= args.num_samples]
    ci = ci_95(fids)
    print(f"\nConfig D {args.dataset.upper()} FID (N={args.num_samples}, {len(fids)} seeds): "
          f"{ci['mean']:.2f} ± {ci['std']:.2f}")
    print(f"  95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
