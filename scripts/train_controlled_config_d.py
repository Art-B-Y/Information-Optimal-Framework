"""Controlled Config D training — full model (path_kl + quality + frozen score backbone).

Step 3A (Session 6). Trains the controlled score model on FashionMNIST for 30
epochs with a pretrained score backbone frozen. 3-seed reliability by default.

Config D definition:
  - path_kl_weight = 0.1  (Girsanov term enabled)
  - quality_weight = 0.01 (feature-matching loss enabled)
  - freeze_score_model = True (backbone frozen, only control policy trains)
  - use_lr_schedule = True

Usage:
    python scripts/train_controlled_config_d.py --score-ckpt checkpoints/score_fmnist_v2/score_epoch_0050.pt
    python scripts/train_controlled_config_d.py --score-ckpt <ckpt> --epochs 30 --seed 42
    python scripts/train_controlled_config_d.py --score-ckpt <ckpt> --all-seeds  # seeds 42, 123, 7
    python scripts/train_controlled_config_d.py --score-ckpt <ckpt> --use-conv-control --epochs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _eval_config_d_fid(ckpt_dir: Path, device: str | None, num_samples: int = 5000,
                        nfe: int = 100, seed: int = 42, use_conv_control: bool = False) -> dict:
    """Load final Config D checkpoint and compute FID with controlled SDE."""
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

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(dev)
    sd = state.get("score_state_dict") or state.get("model_state_dict")
    if sd:
        model.load_state_dict(sd, strict=True)
    model.eval()

    # Detect ConvControl from checkpoint keys or explicit flag
    ctrl_cfg_dict = state.get("control_config", {})
    is_conv = use_conv_control or ("base_channels" in ctrl_cfg_dict)
    if is_conv:
        cc = ctrl_cfg_dict
        conv_cfg = ConvControlConfig(
            in_channels=cc.get("in_channels", 1),
            base_channels=cc.get("base_channels", 64),
            num_blocks=cc.get("num_blocks", 3),
            time_embed_dim=cc.get("time_embed_dim", 128),
            dropout=cc.get("dropout", 0.0),
        )
        ctrl_policy = build_conv_control_policy(conv_cfg, device=dev)
    else:
        ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=256, depth=3, time_embedding_dim=16)
        ctrl_policy = build_control_policy(ctrl_cfg, device=dev, image_shape=(1, 28, 28))
    ctrl_sd = state.get("control_state_dict")
    if ctrl_sd:
        ctrl_policy.load_state_dict(ctrl_sd, strict=True)
    ctrl_policy.eval()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=nfe,
                             sigma_min=0.01, sigma_max=1.0)
    eval_cfg = EvaluationConfig(dataset_name="fashionmnist", num_samples=num_samples,
                                batch_size=64, device=str(dev), seed=seed)
    try:
        results = evaluate_sampler(model, ctrl_policy, sde_cfg, eval_cfg)
        return {"fid": results["fid"], "inception_score_mean": results["inception_score_mean"],
                "nfe_per_sample": nfe, "num_samples_eval": num_samples}
    except Exception as e:
        return {"fid_error": str(e)}


def run_one_seed(score_ckpt: str, epochs: int, seed: int, device: str | None,
                 num_samples_eval: int = 5000, use_conv_control: bool = False) -> dict:
    import torch
    from its.training.controlled_score_training import ControlledScoreTrainingConfig, train_controlled_score
    from its.models import ScoreUNetConfig
    from its.controllers import ControlConfig
    from its.controllers.neural_control import ConvControlConfig
    from its.sde import ScoreSDEConfig

    torch.manual_seed(seed)

    if use_conv_control:
        run_tag = f"controlled_conv_seed{seed}"
        lr = 5e-4
        lr_min = 5e-7
        ema_decay = 0.9999
        batch_size = 128
    else:
        run_tag = f"controlled_config_d_seed{seed}"
        lr = 2e-4
        lr_min = 1e-6
        ema_decay = 0.999
        batch_size = 32

    ckpt_dir = ROOT / "checkpoints" / run_tag
    jsonl_log = ROOT / "logs" / f"{run_tag}.jsonl"
    jsonl_log.parent.mkdir(parents=True, exist_ok=True)

    cfg = ControlledScoreTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        dataset_name="fashionmnist",
        device=device,
        lr=lr,
        control_weight=0.01,
        path_kl_weight=0.1,        # Config D: Girsanov term
        quality_weight=0.01,       # Config D: feature-matching
        grad_clip=1.0,
        log_interval=100,
        nan_tolerance=10,
        jsonl_log=str(jsonl_log),
        checkpoint_dir=str(ckpt_dir),
        resume_latest=True,
        save_interval=5,
        log_dir=str(ROOT / "logs"),
        run_name=run_tag,
        use_lr_schedule=True,
        lr_min=lr_min,
        ema_decay=ema_decay,
        score_backbone_ckpt=score_ckpt,
        freeze_score_model=True,   # Config D: freeze backbone
        mixed_precision=True,
        dataset_subset_size=5000,
        sde=ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=10, clamp=5.0, control_weight=1.0),
        model=ScoreUNetConfig(
            in_channels=1,
            base_channels=64,
            channel_mults=(1, 2, 2),
            use_time_embedding=True,
            time_embed_dim=128,
        ),
        use_conv_control=use_conv_control,
        conv_control=ConvControlConfig(
            in_channels=1,
            base_channels=64,
            num_blocks=3,
            time_embed_dim=128,
            dropout=0.0,
        ) if use_conv_control else None,
        control=ControlConfig(
            state_dim=784,
            hidden_dim=256,
            depth=3,
            time_embedding_dim=16,
        ),
    )

    tag = "ConvControl" if use_conv_control else "Config D"
    print(f"  {tag} seed={seed}: {epochs} epochs, backbone={score_ckpt}")
    metrics = train_controlled_score(cfg)

    # Post-training FID evaluation
    print(f"  Evaluating FID for {tag} seed={seed} (N={num_samples_eval})...", end=" ", flush=True)
    fid_results = _eval_config_d_fid(ckpt_dir, device, num_samples=num_samples_eval, seed=seed,
                                      use_conv_control=use_conv_control)
    if "fid" in fid_results:
        print(f"FID={fid_results['fid']:.2f}")
    else:
        print(f"SKIPPED ({fid_results.get('fid_error', '?')})")

    return {"seed": seed, **metrics, **fid_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True,
                        help="Path to trained FashionMNIST score model checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Single seed to run (overridden by --all-seeds)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run seeds 42, 123, 7 for reliability analysis")
    parser.add_argument("--num-samples-eval", type=int, default=5000,
                        help="N for post-training FID evaluation")
    parser.add_argument("--use-conv-control", action="store_true",
                        help="Use ConvControlPolicy (lr=5e-4, EMA=0.9999, batch=128) instead of MLP")
    args = parser.parse_args()

    seeds = [42, 123, 7] if args.all_seeds else [args.seed]
    results = []

    tag = "ConvControl" if args.use_conv_control else "Config D"
    print(f"{tag} controlled training: epochs={args.epochs}, seeds={seeds}")

    for seed in seeds:
        result = run_one_seed(args.score_ckpt, args.epochs, seed, args.device,
                              num_samples_eval=args.num_samples_eval,
                              use_conv_control=args.use_conv_control)
        results.append(result)
        fid_str = f", FID={result['fid']:.2f}" if "fid" in result else ""
        print(f"  Seed {seed} done: dsm={result.get('dsm_loss', float('nan')):.4f}{fid_str}")

    # Reliability summary: mean +/- std of DSM losses and FID across seeds
    if len(results) > 1:
        import statistics
        dsm_vals = [r.get("dsm_loss", float("nan")) for r in results]
        finite_dsm = [v for v in dsm_vals if v == v]
        fid_vals = [r.get("fid") for r in results if r.get("fid") is not None]
        if finite_dsm:
            mean_dsm = statistics.mean(finite_dsm)
            std_dsm = statistics.stdev(finite_dsm) if len(finite_dsm) > 1 else 0.0
            print(f"\nReliability: DSM = {mean_dsm:.4f} ± {std_dsm:.4f} over {len(finite_dsm)} seeds")
            for r in results:
                r["reliability_dsm_mean"] = mean_dsm
                r["reliability_dsm_std"] = std_dsm
        if fid_vals:
            mean_fid = statistics.mean(fid_vals)
            std_fid = statistics.stdev(fid_vals) if len(fid_vals) > 1 else 0.0
            print(f"Reliability: FID  = {mean_fid:.2f} ± {std_fid:.2f} over {len(fid_vals)} seeds")
            for r in results:
                r["reliability_fid_mean"] = mean_fid
                r["reliability_fid_std"] = std_fid

    out_name = "controlled_conv_results.json" if args.use_conv_control else "controlled_config_d_results.json"
    out_path = ROOT / "data" / "results" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config_tag = "D_conv" if args.use_conv_control else "D"
    out_path.write_text(json.dumps({"config": config_tag, "epochs": args.epochs, "runs": results}, indent=2))
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
