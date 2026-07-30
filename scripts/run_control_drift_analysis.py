"""Control drift analysis — measure how much the control policy shifts the SDE trajectory.

Step 6 (Session 6). Analyzes the controlled vs uncontrolled SDE trajectories to
quantify the effect of the control policy:
  - Mean control magnitude per time step
  - Cosine similarity between score field and control field
  - Score/control ratio over time

Saves results to data/results/control_drift_analysis.json.

Usage:
    python scripts/run_control_drift_analysis.py --model-ckpt <score_ckpt> --control-ckpt <ctrl_ckpt>
    python scripts/run_control_drift_analysis.py --from-jsonl logs/controlled_config_d_seed42.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def analyze_from_jsonl(jsonl_path: Path) -> dict:
    """Extract control drift metrics from a controlled training JSONL log."""
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            # Support both old keys (cos_sim/ctrl_mag/score_mag) and new keys from controlled training
            has_keys = (
                ("cos_sim" in r and "ctrl_mag" in r and "score_mag" in r) or
                ("cos_sim_score_ctrl" in r and "ctrl_drift_mag" in r and "score_drift_mag" in r)
            )
            if has_keys:
                # Normalize to canonical key names
                norm = dict(r)
                if "cos_sim_score_ctrl" in r:
                    norm["cos_sim"] = r["cos_sim_score_ctrl"]
                    norm["ctrl_mag"] = r["ctrl_drift_mag"]
                    norm["score_mag"] = r["score_drift_mag"]
                    norm["ratio"] = r.get("ctrl_score_mag_ratio",
                                         r["ctrl_drift_mag"] / max(r["score_drift_mag"], 1e-8))
                records.append(norm)
        except json.JSONDecodeError:
            pass

    if not records:
        return {"error": "No diagnostic records found in JSONL"}

    cos_sims = [r["cos_sim"] for r in records]
    ctrl_mags = [r["ctrl_mag"] for r in records]
    score_mags = [r["score_mag"] for r in records]
    ratios = [r.get("ratio", r["ctrl_mag"] / max(r["score_mag"], 1e-8)) for r in records]

    import statistics
    result = {
        "source": str(jsonl_path),
        "n_records": len(records),
        "cos_sim_mean": statistics.mean(cos_sims),
        "cos_sim_std": statistics.stdev(cos_sims) if len(cos_sims) > 1 else 0.0,
        "ctrl_mag_mean": statistics.mean(ctrl_mags),
        "ctrl_mag_final": ctrl_mags[-1] if ctrl_mags else None,
        "score_mag_mean": statistics.mean(score_mags),
        "ctrl_score_ratio_mean": statistics.mean(ratios),
        "ctrl_score_ratio_final": ratios[-1] if ratios else None,
        "time_series": {
            "epochs": [r.get("epoch") for r in records],
            "cos_sim": cos_sims,
            "ctrl_mag": ctrl_mags,
            "score_mag": score_mags,
            "ratio": ratios,
        },
    }
    return result


def analyze_from_checkpoints(score_ckpt: str, control_ckpt: str, device: str) -> dict:
    """Load checkpoints and sample trajectories to measure control drift."""
    import torch
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy
    from its.sde import ScoreSDESimulator, ScoreSDEConfig

    dev = torch.device(device)

    # Load score model
    state = torch.load(score_ckpt, map_location="cpu", weights_only=False)
    model_cfg_dict = state.get("model_config") or state.get("score_config")
    if model_cfg_dict:
        model_cfg = ScoreUNetConfig(**model_cfg_dict)
    else:
        model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64,
                                    channel_mults=(1, 2, 2), use_time_embedding=True, time_embed_dim=128)
    score_model = build_score_model(model_cfg).to(dev)
    score_sd = state.get("ema_state_dict") or state.get("score_state_dict") or state.get("model_state_dict") or state
    score_model.load_state_dict(score_sd, strict=True)
    score_model.eval()

    # Load control policy
    ctrl_state = torch.load(control_ckpt, map_location="cpu", weights_only=False)
    ctrl_cfg_dict = ctrl_state.get("control_config")
    if ctrl_cfg_dict:
        ctrl_cfg = ControlConfig(**ctrl_cfg_dict)
    else:
        ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=256, depth=3, time_embedding_dim=16)
    ctrl_policy = build_control_policy(ctrl_cfg).to(dev)
    ctrl_sd = ctrl_state.get("control_state_dict") or ctrl_state
    ctrl_policy.load_state_dict(ctrl_sd, strict=True)
    ctrl_policy.eval()

    # Sample one mini-batch and measure control drift
    N, C, H, W = 32, 1, 28, 28
    with torch.no_grad():
        x = torch.randn(N, C, H, W, device=dev)
        t_vals = torch.linspace(0, 1, 20, device=dev)
        ctrl_mags, score_mags, cos_sims = [], [], []
        for t_val in t_vals:
            t = t_val.expand(N, 1)
            score = score_model(x, t)
            ctrl = ctrl_policy(x.flatten(1), t)
            ctrl_reshaped = ctrl.view(N, C, H, W)
            ctrl_mag = float(ctrl_reshaped.norm(dim=[1, 2, 3]).mean())
            score_mag = float(score.norm(dim=[1, 2, 3]).mean())
            cos_sim = float(
                torch.nn.functional.cosine_similarity(
                    ctrl_reshaped.flatten(1), score.flatten(1), dim=1
                ).mean()
            )
            ctrl_mags.append(ctrl_mag)
            score_mags.append(score_mag)
            cos_sims.append(cos_sim)

    result = {
        "source": f"checkpoints: score={score_ckpt}, control={control_ckpt}",
        "n_time_steps": len(t_vals),
        "ctrl_mag_mean": sum(ctrl_mags) / len(ctrl_mags),
        "score_mag_mean": sum(score_mags) / len(score_mags),
        "cos_sim_mean": sum(cos_sims) / len(cos_sims),
        "ctrl_score_ratio_mean": sum(c / max(s, 1e-8) for c, s in zip(ctrl_mags, score_mags)) / len(ctrl_mags),
        "time_series": {
            "t": [float(t) for t in t_vals.tolist()],
            "ctrl_mag": ctrl_mags,
            "score_mag": score_mags,
            "cos_sim": cos_sims,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-jsonl", type=str, help="JSONL log from controlled training")
    group.add_argument("--model-ckpt", type=str, help="Score model checkpoint")
    parser.add_argument("--control-ckpt", type=str, help="Control policy checkpoint (with --model-ckpt)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.from_jsonl:
        result = analyze_from_jsonl(Path(args.from_jsonl))
    else:
        if not args.control_ckpt:
            raise ValueError("--control-ckpt required with --model-ckpt")
        result = analyze_from_checkpoints(args.model_ckpt, args.control_ckpt, args.device)

    print(f"Control drift analysis: {result.get('n_records', result.get('n_time_steps', '?'))} samples")
    print(f"  Ctrl mag (mean):  {result.get('ctrl_mag_mean', 'N/A'):.4f}")
    print(f"  Score mag (mean): {result.get('score_mag_mean', 'N/A'):.4f}")
    print(f"  Cos sim (mean):   {result.get('cos_sim_mean', 'N/A'):.4f}")
    print(f"  Ctrl/score ratio: {result.get('ctrl_score_ratio_mean', 'N/A'):.4f}")

    out_path = ROOT / "data" / "results" / "control_drift_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
