"""Step 1C: Controller collapse trajectory plot.

Loads the controlled_conv_seed42 checkpoints (epochs 5, 10, ..., 50),
runs a forward pass on a fixed 16-sample batch, and computes:
  - mean ||u_theta(x, t)|| at t=0.5
  - mean ||score(x, t)|| at t=0.5
  - ratio control_magnitude / score_magnitude

Saves:
  data/paper_figures/controller_collapse_trajectory.{png,pdf}
  data/results/collapse_trajectory_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from its.models import ScoreUNetConfig, build_score_model
from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy, ConvControlPolicy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    score_ckpt = ROOT / "checkpoints" / "score_fmnist_v2" / "score_best.pt"
    ctrl_dir = ROOT / "checkpoints" / "controlled_conv_seed42"

    # Load score model
    score_state = torch.load(score_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                 use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(device)
    sd = score_state.get("score_state_dict") or score_state.get("model_state_dict") or score_state.get("model")
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Fixed batch
    ds = datasets.FashionMNIST(
        root=str(ROOT / "data" / "tensors"), train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    x_fixed, _ = next(iter(loader))
    x_fixed = x_fixed.to(device)
    t_val = 0.5
    sigma_val = 0.5
    sigma = torch.full((x_fixed.shape[0], 1, 1, 1), sigma_val, device=device)
    t_batch = torch.full((x_fixed.shape[0], 1), t_val, device=device)

    with torch.no_grad():
        score_vecs = model(x_fixed, sigma)
        score_mag = score_vecs.reshape(score_vecs.shape[0], -1).norm(dim=1).mean().item()

    epochs = list(range(5, 55, 5))  # 5, 10, ..., 50
    metrics = []

    for epoch in epochs:
        ckpt_path = ctrl_dir / f"controlled_epoch_{epoch:04d}.pt"
        if not ckpt_path.exists():
            print(f"  Epoch {epoch}: checkpoint not found, skipping")
            continue

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ctrl_cfg_dict = state.get("control_config", {})
        conv_cfg = ConvControlConfig(
            in_channels=ctrl_cfg_dict.get("in_channels", 1),
            base_channels=ctrl_cfg_dict.get("base_channels", 64),
            num_blocks=ctrl_cfg_dict.get("num_blocks", 3),
            time_embed_dim=ctrl_cfg_dict.get("time_embed_dim", 128),
            dropout=ctrl_cfg_dict.get("dropout", 0.0),
        )
        ctrl = build_conv_control_policy(conv_cfg, device=device)
        ctrl.load_state_dict(state["control_state_dict"], strict=True)
        ctrl.eval()

        with torch.no_grad():
            ctrl_vecs = ctrl(x_fixed, t_batch)
        ctrl_mag = ctrl_vecs.reshape(ctrl_vecs.shape[0], -1).norm(dim=1).mean().item()
        ratio = ctrl_mag / (score_mag + 1e-15)

        metrics.append({
            "epoch": epoch,
            "control_magnitude": ctrl_mag,
            "score_magnitude": score_mag,
            "ratio_ctrl_over_score": ratio,
        })
        print(f"  Epoch {epoch:2d}: ||u||={ctrl_mag:.4f}  ||s||={score_mag:.4f}  ratio={ratio:.6f} ({ratio*100:.4f}%)")

    # Save metrics JSON
    out_dir = ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "collapse_trajectory_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics -> {json_path}")

    # Plot
    fig_dir = ROOT / "data" / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep_vals = [m["epoch"] for m in metrics]
        ratio_vals = [m["ratio_ctrl_over_score"] * 100 for m in metrics]
        ctrl_vals = [m["control_magnitude"] for m in metrics]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(ep_vals, ratio_vals, "o-", color="#d62728", linewidth=2, markersize=6)
        axes[0].axhline(0.1, color="gray", linestyle="--", alpha=0.6, label="0.1% threshold")
        axes[0].set_xlabel("Training Epoch")
        axes[0].set_ylabel("||u|| / ||score|| (%)")
        axes[0].set_title("Control / Score Magnitude Ratio")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(ep_vals, ctrl_vals, "o-", color="#1f77b4", linewidth=2, markersize=6)
        axes[1].set_xlabel("Training Epoch")
        axes[1].set_ylabel("Mean ||u_theta|| (log scale)")
        axes[1].set_title("Controller Output Magnitude Collapse")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig_path = fig_dir / f"controller_collapse_trajectory.{ext}"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure -> {fig_path}")
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")


if __name__ == "__main__":
    main()
