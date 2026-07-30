"""Run 7-ablation study for ITS controlled score model.

Step 5 (Session 6). Evaluates 7 ablation configurations on FashionMNIST.
Each run uses N=2048 samples, seed=42, 5 epochs with frozen score backbone.

Ablation configs:
  B: path_kl=0.0, quality=0.0  (pure control energy)
  C: path_kl=0.1, quality=0.0  (Girsanov without feature-matching)
  D: path_kl=0.1, quality=0.01 (full model, MLP controller)
  F: path_kl=0.1, quality=0.1  (stronger quality weight)
  conv: same as D but with ConvControlPolicy instead of MLP (Ablation 5)
  no-EMA: same as D but ema_decay=0.0
  no-freeze: same as D but freeze_score_model=False

Usage:
    python scripts/run_ablation_study.py --score-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt
    python scripts/run_ablation_study.py --score-ckpt <ckpt> --ablations B C D --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ABLATION_CONFIGS = {
    "B":        {"path_kl_weight": 0.0, "quality_weight": 0.0,  "freeze_score_model": True,  "ema_decay": 0.999, "use_conv_control": False},
    "C":        {"path_kl_weight": 0.1, "quality_weight": 0.0,  "freeze_score_model": True,  "ema_decay": 0.999, "use_conv_control": False},
    "D":        {"path_kl_weight": 0.1, "quality_weight": 0.01, "freeze_score_model": True,  "ema_decay": 0.999, "use_conv_control": False},
    "F":        {"path_kl_weight": 0.1, "quality_weight": 0.1,  "freeze_score_model": True,  "ema_decay": 0.999, "use_conv_control": False},
    "conv":     {"path_kl_weight": 0.1, "quality_weight": 0.01, "freeze_score_model": True,  "ema_decay": 0.999, "use_conv_control": True},
    "no-EMA":   {"path_kl_weight": 0.1, "quality_weight": 0.01, "freeze_score_model": True,  "ema_decay": 0.0,   "use_conv_control": False},
    "no-freeze":{"path_kl_weight": 0.1, "quality_weight": 0.01, "freeze_score_model": False, "ema_decay": 0.999, "use_conv_control": False},
}


def _eval_ablation_fid(ckpt_dir: Path, epochs: int, name: str, num_samples: int,
                       device_str: str, use_conv_control: bool = False) -> dict:
    """Load saved ablation checkpoint and compute FID with controlled SDE."""
    import torch
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy
    from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy
    from its.eval import EvaluationConfig
    from its.eval.evaluator import evaluate_sampler
    from its.sde import ScoreSDEConfig

    candidates = sorted(ckpt_dir.glob("controlled_epoch_*.pt"))
    if not candidates:
        return {"fid_error": "no checkpoint found"}
    ckpt_path = candidates[-1]

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(device)
    sd = state.get("score_state_dict") or state.get("model_state_dict")
    if sd:
        model.load_state_dict(sd, strict=True)
    model.eval()

    ctrl_sd = state.get("control_state_dict") or {}
    if use_conv_control:
        ctrl_cfg = ConvControlConfig(in_channels=1)
        control_policy = build_conv_control_policy(ctrl_cfg, device=device)
    else:
        ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=256, depth=3, time_embedding_dim=16)
        control_policy = build_control_policy(ctrl_cfg, device=device, image_shape=(1, 28, 28))
    if ctrl_sd:
        control_policy.load_state_dict(ctrl_sd, strict=False)
    control_policy.eval()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=100,
                             sigma_min=0.01, sigma_max=1.0)
    eval_cfg = EvaluationConfig(dataset_name="fashionmnist", num_samples=num_samples,
                                batch_size=64, device=str(device), seed=42)
    try:
        results = evaluate_sampler(model, control_policy, sde_cfg, eval_cfg)
        return {"fid": results["fid"], "inception_score_mean": results["inception_score_mean"],
                "nfe_per_sample": 100}
    except Exception as e:
        return {"fid_error": str(e)}


def run_ablation(name: str, overrides: dict, score_ckpt: str, epochs: int, device: str | None,
                 num_samples: int) -> dict:
    import torch
    from its.training.controlled_score_training import ControlledScoreTrainingConfig, train_controlled_score
    from its.models import ScoreUNetConfig
    from its.controllers import ControlConfig
    from its.sde import ScoreSDEConfig

    torch.manual_seed(42)

    ckpt_dir = ROOT / "checkpoints" / f"ablation_{name.replace('-', '_')}"
    jsonl_log = ROOT / "logs" / f"ablation_{name.replace('-', '_')}.jsonl"
    jsonl_log.parent.mkdir(parents=True, exist_ok=True)

    cfg = ControlledScoreTrainingConfig(
        epochs=epochs,
        batch_size=32,
        dataset_name="fashionmnist",
        device=device,
        lr=2e-4,
        control_weight=0.01,
        grad_clip=1.0,
        log_interval=100,
        nan_tolerance=10,
        jsonl_log=str(jsonl_log),
        checkpoint_dir=str(ckpt_dir),
        save_interval=1,  # save every epoch so final checkpoint is available for eval
        log_dir=str(ROOT / "logs"),
        run_name=f"ablation_{name}",
        use_lr_schedule=True,
        lr_min=1e-6,
        score_backbone_ckpt=score_ckpt,
        mixed_precision=False,
        dataset_subset_size=5000,
        sde=ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=10, clamp=5.0, control_weight=1.0),
        model=ScoreUNetConfig(
            in_channels=1,
            base_channels=64,
            channel_mults=(1, 2, 2),
            use_time_embedding=True,
            time_embed_dim=128,
        ),
        control=ControlConfig(
            state_dim=784,
            hidden_dim=256,
            depth=3,
            time_embedding_dim=16,
        ),
        # Apply ablation overrides
        **overrides,
    )

    device_str = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"  Ablation {name}: epochs={epochs}, overrides={list(overrides.keys())}")
    try:
        metrics = train_controlled_score(cfg)
        # Post-training FID evaluation
        print(f"  Evaluating FID for ablation {name} (N={num_samples})...", end=" ", flush=True)
        use_conv = overrides.get("use_conv_control", False)
        fid_results = _eval_ablation_fid(ckpt_dir, epochs, name, num_samples, device_str,
                                         use_conv_control=use_conv)
        if "fid" in fid_results:
            print(f"FID={fid_results['fid']:.2f}")
        else:
            print(f"SKIPPED ({fid_results.get('fid_error', '?')})")
        clean_overrides = {k: v for k, v in overrides.items() if k != "use_conv_control"}
        return {"name": name, "status": "ok", **metrics, **fid_results, **clean_overrides}
    except Exception as e:
        print(f"  FAILED: {e}")
        return {"name": name, "status": "failed", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True,
                        help="Path to trained FashionMNIST score model checkpoint")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per ablation (default 5 for speed)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ablations", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        choices=list(ABLATION_CONFIGS.keys()),
                        help="Which ablations to run (default: all 7)")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs, 256 samples for smoke test")
    args = parser.parse_args()

    epochs = 2 if args.quick else args.epochs
    num_samples = 256 if args.quick else args.num_samples

    print(f"Ablation study: {len(args.ablations)} configs, epochs={epochs}, N={num_samples}")
    results = []

    for name in args.ablations:
        overrides = ABLATION_CONFIGS[name]
        result = run_ablation(name, overrides, args.score_ckpt, epochs, args.device, num_samples)
        results.append(result)

    out_path = ROOT / "data" / "results" / "ablation_study.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"ablations": results, "epochs": epochs, "num_samples": num_samples}, indent=2))
    print(f"\nAblation study complete: {len(results)} configs")
    print(f"Results -> {out_path}")

    # Print summary
    print(f"\n{'Config':<12} {'Status':<8} {'DSM Loss':>10} {'Ctrl Energy':>12}")
    print("-" * 44)
    for r in results:
        dsm = r.get("dsm_loss", float("nan"))
        ctrl = r.get("control_energy", float("nan"))
        print(f"{r['name']:<12} {r.get('status','?'):<8} {dsm:>10.4f} {ctrl:>12.6f}")


if __name__ == "__main__":
    main()
