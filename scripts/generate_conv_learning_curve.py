"""Generate ConvControl vs MLP learning curve comparison (Session 9 Step 2D).

Reads checkpoint FID trajectories from:
  - data/results/conv_checkpoint_fid_trajectory.json (ConvControl, from eval_conv_checkpoints.py)
  - Ablation study result for MLP Config D at epoch 5 (FID=326)
  - controlled_config_d_multiseed.json for MLP endpoint at epoch 60 (FID=327.47)

Produces:
  - data/paper_figures/conv_vs_mlp_learning_curve.{png,pdf}

Usage:
    python scripts/generate_conv_learning_curve.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "data" / "paper_figures"


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure generation")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load ConvControl trajectory
    conv_traj_path = ROOT / "data" / "results" / "conv_checkpoint_fid_trajectory.json"
    conv_epochs, conv_fids = [], []
    if conv_traj_path.exists():
        records = json.loads(conv_traj_path.read_text())
        for r in records:
            if "fid" in r and "epoch" in r:
                conv_epochs.append(r["epoch"])
                conv_fids.append(r["fid"])

    # MLP reference points
    mlp_5ep_fid = 326.0  # ablation_conv (5-epoch ablation, same as MLP)
    mlp_60ep_fid = 327.47  # Config D 3-seed mean, 60 epochs

    # DDPM reference
    ddpm_fid = 236.28  # at NFE=100, 3-seed, 2048 samples

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if conv_epochs:
        ax.plot(conv_epochs, conv_fids, "b-o", label="ConvControl (50 ep)", linewidth=1.8,
                markersize=5, zorder=3)

    # MLP reference points
    ax.scatter([5, 60], [mlp_5ep_fid, mlp_60ep_fid], color="orange", s=80, zorder=4,
               label="MLP Config D")
    if len([5, 60]) > 1:
        ax.plot([5, 60], [mlp_5ep_fid, mlp_60ep_fid], "orange", linestyle="--", linewidth=1.2, alpha=0.7)

    # DDPM reference line
    ax.axhline(y=ddpm_fid, color="red", linestyle=":", linewidth=1.5,
               label=f"DDPM NFE=100 (FID={ddpm_fid:.1f})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("FID (↓ better)", fontsize=12)
    ax.set_title("ConvControl vs MLP: FID Trajectory (FashionMNIST, NFE=100)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if not conv_epochs:
        ax.text(0.5, 0.5, "ConvControl training in progress\n(no checkpoints evaluated yet)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray",
                style="italic")
        ax.set_xlim(0, 55)
        ax.set_ylim(190, 380)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"conv_vs_mlp_learning_curve.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: conv_vs_mlp_learning_curve.{{png,pdf}}")


if __name__ == "__main__":
    main()
