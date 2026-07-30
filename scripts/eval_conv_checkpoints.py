"""Evaluate ConvControl checkpoints at each saved epoch.

Reads all controlled_epoch_NNNN.pt files in the checkpoint dir and computes
quick FID (512 samples) to track the training trajectory. Skips epochs already
evaluated (results file is updated incrementally).

Usage:
    python scripts/eval_conv_checkpoints.py
    python scripts/eval_conv_checkpoints.py --ckpt-dir checkpoints/controlled_conv_seed42 --num-samples 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def eval_checkpoint(ckpt_path: Path, device: str, num_samples: int, nfe: int) -> dict:
    import torch
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy
    from its.eval import EvaluationConfig
    from its.eval.evaluator import evaluate_sampler
    from its.sde import ScoreSDEConfig

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = state.get("epoch", -1)

    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(dev)
    sd = state.get("score_state_dict") or state.get("model_state_dict")
    if sd:
        model.load_state_dict(sd, strict=True)
    model.eval()

    ctrl_cfg_dict = state.get("control_config", {})
    conv_cfg = ConvControlConfig(
        in_channels=ctrl_cfg_dict.get("in_channels", 1),
        base_channels=ctrl_cfg_dict.get("base_channels", 64),
        num_blocks=ctrl_cfg_dict.get("num_blocks", 3),
        time_embed_dim=ctrl_cfg_dict.get("time_embed_dim", 128),
        dropout=ctrl_cfg_dict.get("dropout", 0.0),
    )
    ctrl_policy = build_conv_control_policy(conv_cfg, device=dev)
    ctrl_sd = state.get("control_state_dict")
    if ctrl_sd:
        ctrl_policy.load_state_dict(ctrl_sd, strict=True)
    ctrl_policy.eval()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=nfe,
                             sigma_min=0.01, sigma_max=1.0)
    eval_cfg = EvaluationConfig(dataset_name="fashionmnist", num_samples=num_samples,
                                batch_size=64, device=str(dev), seed=42)
    try:
        results = evaluate_sampler(model, ctrl_policy, sde_cfg, eval_cfg)
        return {"epoch": epoch, "fid": results["fid"], "nfe": nfe, "num_samples": num_samples,
                "ckpt": ckpt_path.name}
    except Exception as e:
        return {"epoch": epoch, "error": str(e), "ckpt": ckpt_path.name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir",
                        default=str(ROOT / "checkpoints" / "controlled_conv_seed42"))
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out",
                        default=str(ROOT / "data" / "results" / "conv_checkpoint_fid_trajectory.json"))
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out)

    # Load existing results
    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            existing = {r["ckpt"]: r for r in json.loads(out_path.read_text())}
        except Exception:
            pass

    checkpoints = sorted(ckpt_dir.glob("controlled_epoch_*.pt"))
    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    results = list(existing.values())
    existing_names = set(existing.keys())

    for ckpt_path in checkpoints:
        if ckpt_path.name in existing_names:
            print(f"  Skip {ckpt_path.name} (already evaluated)")
            continue
        print(f"  Evaluating {ckpt_path.name}...", end=" ", flush=True)
        r = eval_checkpoint(ckpt_path, args.device, args.num_samples, args.nfe)
        if "fid" in r:
            print(f"FID={r['fid']:.2f} (epoch {r['epoch']})")
        else:
            print(f"ERROR: {r.get('error', '?')}")
        results.append(r)
        results.sort(key=lambda x: x.get("epoch", 0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    if results:
        best = min((r for r in results if "fid" in r), key=lambda x: x["fid"], default=None)
        print(f"\nBest: epoch {best['epoch']}, FID={best['fid']:.2f}" if best else "\nNo valid results")
        print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
