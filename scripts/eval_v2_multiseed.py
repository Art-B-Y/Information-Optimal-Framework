"""Paper-grade evaluation of v2 controlled training runs (Step 5D, Session 10).

Runs 3-seed FID evaluation (N=5000) on a v2 checkpoint and reports 95% CI.
Supports runs 5A (frozen backbone), 5B (joint finetune), 5C (no path-KL).

Usage:
    python scripts/eval_v2_multiseed.py --run-id 5a
    python scripts/eval_v2_multiseed.py --run-id 5b
    python scripts/eval_v2_multiseed.py --run-id 5c
    # Custom checkpoint:
    python scripts/eval_v2_multiseed.py --run-id 5a --ctrl-ckpt checkpoints/controlled_v2_5a_seed42/controlled_last.pt
    # Quick (N=512 smoke test):
    python scripts/eval_v2_multiseed.py --run-id 5a --quick
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

_RUN_CKPT_DIRS = {
    "5a": ROOT / "checkpoints" / "controlled_v2_5a_seed42",
    "5b": ROOT / "checkpoints" / "controlled_v2_5b_joint",
    "5c": ROOT / "checkpoints" / "controlled_v2_5c_nopkL",
    "5a_pilot": ROOT / "checkpoints" / "controlled_v2_5a_pilot",
}

_SCORE_CKPT = ROOT / "checkpoints" / "score_fmnist_v2" / "score_best.pt"


def ci_95(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"lower": mean, "upper": mean, "mean": mean, "std": 0.0}
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    half_width = 1.96 * std / math.sqrt(n)
    return {"lower": mean - half_width, "upper": mean + half_width, "mean": mean, "std": std}


def eval_one(
    score_model: torch.nn.Module,
    ctrl_policy: torch.nn.Module,
    sde_cfg,
    num_samples: int,
    seed: int,
    nfe: int,
    device: torch.device,
) -> dict:
    from its.eval.evaluator import evaluate_sampler
    from its.eval import EvaluationConfig
    from its.sde import ScoreSDEConfig

    nfe_cfg = ScoreSDEConfig(
        beta_min=sde_cfg.beta_min,
        beta_max=sde_cfg.beta_max,
        num_steps=nfe,
        sigma_min=sde_cfg.sigma_min,
        sigma_max=sde_cfg.sigma_max,
    )
    eval_cfg = EvaluationConfig(
        dataset_name="fashionmnist",
        num_samples=num_samples,
        batch_size=64,
        device=str(device),
        seed=seed,
    )
    result = evaluate_sampler(score_model, ctrl_policy, nfe_cfg, eval_cfg)
    return {
        "fid": result["fid"],
        "inception_score_mean": result.get("inception_score_mean"),
        "nfe": nfe,
        "num_samples": num_samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, choices=["5a", "5b", "5c", "5a_pilot"],
                        help="Which v2 training run to evaluate")
    parser.add_argument("--ctrl-ckpt", default=None,
                        help="Path to controlled_last.pt (overrides --run-id default)")
    parser.add_argument("--score-ckpt", default=str(_SCORE_CKPT),
                        help="Path to score model checkpoint")
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="N=512 smoke test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_samples = 512 if args.quick else args.num_samples

    # Resolve checkpoint path
    ctrl_ckpt_path: Path
    if args.ctrl_ckpt:
        ctrl_ckpt_path = Path(args.ctrl_ckpt)
    else:
        ckpt_dir = _RUN_CKPT_DIRS[args.run_id]
        ctrl_ckpt_path = ckpt_dir / "controlled_last.pt"

    if not ctrl_ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ctrl_ckpt_path}")
        print(f"  Run 'python scripts/train_redesigned_v2.py --run {args.run_id}' first.")
        sys.exit(1)

    out_path = ROOT / "data" / "results" / f"controlled_v2_{args.run_id}_multiseed.json"

    print(f"=== eval_v2_multiseed: Run {args.run_id.upper()} ===")
    print(f"Ctrl ckpt : {ctrl_ckpt_path}")
    print(f"Score ckpt: {args.score_ckpt}")
    print(f"N={num_samples}, NFE={args.nfe}, seeds={SEEDS}")

    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy
    from its.sde import ScoreSDEConfig

    # Load score model
    score_state = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    score_cfg_dict = score_state.get("score_config") or score_state.get("model_config")
    if score_cfg_dict:
        score_cfg = ScoreUNetConfig(**score_cfg_dict)
    else:
        score_cfg = ScoreUNetConfig(in_channels=1, base_channels=64,
                                    channel_mults=(1, 2, 2), use_time_embedding=True,
                                    time_embed_dim=128)
    score_model = build_score_model(score_cfg).to(device)
    score_sd = (score_state.get("score_state_dict") or
                score_state.get("model_state_dict") or score_state)
    score_model.load_state_dict(score_sd, strict=True)
    score_model.eval()

    # Load v2 controlled checkpoint
    ctrl_state = torch.load(ctrl_ckpt_path, map_location="cpu", weights_only=False)

    # Prefer joint-checkpoint score weights if available (for run 5B)
    if "score_state_dict" in ctrl_state and args.run_id == "5b":
        score_model.load_state_dict(ctrl_state["score_state_dict"], strict=True)
        print("  Using joint-finetuned score weights from ctrl checkpoint.")

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

    ckpt_epoch = ctrl_state.get("epoch", "?")
    print(f"  Loaded ctrl checkpoint (epoch={ckpt_epoch})")

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=args.nfe,
                              sigma_min=0.01, sigma_max=1.0)

    # Resume-aware: load existing results
    results: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            results = existing.get("runs", [])
            print(f"Resuming: {len(results)}/3 seeds already done.")
        except Exception:
            results = []

    done_seeds = {r["seed"] for r in results if r.get("num_samples", 0) >= num_samples}

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"  SKIP seed={seed} (already done)")
            continue
        print(f"  Evaluating seed={seed}...", end=" ", flush=True)
        try:
            entry = eval_one(score_model, ctrl_policy, sde_cfg, num_samples, seed, args.nfe, device)
            print(f"FID={entry['fid']:.2f}")
            results.append(entry)
            fids = [r["fid"] for r in results if r.get("num_samples", 0) >= num_samples]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "run_id": args.run_id,
                "ctrl_ckpt": str(ctrl_ckpt_path),
                "epoch": ckpt_epoch,
                "nfe": args.nfe,
                "num_samples": num_samples,
                "runs": results,
                "ci_95": ci_95(fids),
            }, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")

    fids = [r["fid"] for r in results if r.get("num_samples", 0) >= num_samples]
    if fids:
        ci = ci_95(fids)
        print(f"\nRun {args.run_id.upper()} FID (N={num_samples}, {len(fids)} seeds):")
        print(f"  {ci['mean']:.2f} ± {ci['std']:.2f}  (95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}])")
        print(f"  Per-seed: {[r['fid'] for r in results]}")
        print(f"Saved -> {out_path}")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
