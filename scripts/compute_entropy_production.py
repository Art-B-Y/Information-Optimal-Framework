"""Compute entropy production profiles for controlled vs uncontrolled SDE (Step 4C).

The entropy production proxy at each time step is |u(x,t)|^2 * dt,
where u is the control signal. For the uncontrolled SDE, u=0 everywhere.

Cumulative entropy production shows how irreversibility accumulates during generation.

Usage:
    python scripts/compute_entropy_production.py \
        --score-ckpt checkpoints/score_fmnist_v2/score_best.pt \
        --ctrl-ckpt checkpoints/controlled_config_d_seed42/controlled_last.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def compute_profile(score_model, ctrl_policy, sde_cfg, device, n_trajectories=256):
    from its.sde import ScoreSDEConfig
    import torch.nn.functional as F

    beta_min = sde_cfg.beta_min
    beta_max = sde_cfg.beta_max
    T = sde_cfg.num_steps
    dt = 1.0 / T

    # Batch of trajectories
    x = torch.randn(n_trajectories, 1, 28, 28, device=device)
    flat_dim = 784

    ctrl_energy_per_step = []
    unctrld_energy_per_step = []  # zero by definition

    with torch.no_grad():
        for step_idx in range(T):
            t_frac = 1.0 - step_idx / T  # reverse time: 1 → 0
            sigma = (beta_min + (beta_max - beta_min) * t_frac) ** 0.5

            # Score network output
            sigma_t = torch.tensor([sigma], device=device)
            score = score_model(x, sigma_t.expand(x.shape[0]))

            # Control signal
            x_flat = x.reshape(n_trajectories, flat_dim)
            t_tensor = torch.full((n_trajectories,), t_frac, device=device)
            ctrl = ctrl_policy(x_flat, t_tensor)

            # Entropy production proxy: E[|u|^2] * dt
            ctrl_ep = float(ctrl.pow(2).sum(dim=1).mean().item()) * dt
            ctrl_energy_per_step.append(ctrl_ep)
            unctrld_energy_per_step.append(0.0)

            # SDE step (Euler-Maruyama, no grad needed for profile)
            #
            # Audit 2026-07-15 (C3): the control was computed and tallied into
            # the energy above but NEVER added to the drift, so this walked the
            # UNCONTROLLED trajectory and the resulting "ITS entropy production"
            # profile described the uncontrolled path.  The control term is now
            # integrated, so the trajectory is the one whose energy we report.
            beta_t = beta_min + (beta_max - beta_min) * t_frac
            drift = 0.5 * beta_t * (-x - score)
            noise = torch.randn_like(x) * (beta_t * dt) ** 0.5
            x = x + (drift + ctrl.reshape_as(x)) * dt + noise

    # Cumulative sums
    cumulative_ctrl = []
    cumulative_unctrl = []
    running_c, running_u = 0.0, 0.0
    for c, u in zip(ctrl_energy_per_step, unctrld_energy_per_step):
        running_c += c
        running_u += u
        cumulative_ctrl.append(running_c)
        cumulative_unctrl.append(running_u)

    return {
        "steps": list(range(T)),
        "per_step_ctrl": ctrl_energy_per_step,
        "per_step_uncontrolled": unctrld_energy_per_step,
        "controlled_cumulative": cumulative_ctrl,
        "uncontrolled_cumulative": cumulative_unctrl,
        "total_ctrl_entropy": running_c,
        "total_unctrld_entropy": running_u,
        "n_trajectories": n_trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True)
    parser.add_argument("--ctrl-ckpt", required=True)
    parser.add_argument("--n-trajectories", type=int, default=256)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy
    from its.sde import ScoreSDEConfig

    # Load score model
    score_state = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                 use_time_embedding=True, time_embed_dim=128)
    score_model = build_score_model(model_cfg).to(device)
    score_model.load_state_dict(score_state["model_state_dict"], strict=True)
    score_model.eval()

    # Load control policy
    ctrl_state = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
    ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=256, depth=3, time_embedding_dim=16)
    ctrl_policy = build_control_policy(ctrl_cfg, device=device, image_shape=(1, 28, 28))
    ctrl_policy.load_state_dict(ctrl_state["control_state_dict"], strict=True)
    ctrl_policy.eval()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=args.nfe,
                              sigma_min=0.01, sigma_max=1.0)

    print(f"Computing entropy production profile ({args.n_trajectories} trajectories, {args.nfe} steps)...")
    profile = compute_profile(score_model, ctrl_policy, sde_cfg, device, args.n_trajectories)

    out = ROOT / "data" / "results" / "entropy_production_profile.json"
    out.write_text(json.dumps(profile, indent=2))
    print(f"\nEntropy production profile:")
    print(f"  Total controlled entropy: {profile['total_ctrl_entropy']:.6f}")
    print(f"  Total uncontrolled entropy: 0.0 (by definition)")
    print(f"  Result -> {out}")


if __name__ == "__main__":
    main()
