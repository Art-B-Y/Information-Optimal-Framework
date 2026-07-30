"""Step 1D: Loss term magnitude analysis at epoch 50.

Loads the Session 9 ConvControl checkpoint (epoch 50), runs 50 batches
(batch size 16) and computes mean+std of each loss term.

Saves: data/results/loss_term_magnitudes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from its.models import ScoreUNetConfig, build_score_model
from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy
from its.sde import ScoreSDEConfig
from its.training.controlled_score_training import simulate_path, _build_feature_extractor, _feature_matching_loss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    n_batches = 50
    batch_size = 16

    score_ckpt = ROOT / "checkpoints" / "score_fmnist_v2" / "score_best.pt"
    ctrl_ckpt = ROOT / "checkpoints" / "controlled_conv_seed42" / "controlled_last.pt"

    score_state = torch.load(score_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                 use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(device)
    sd = score_state.get("score_state_dict") or score_state.get("model_state_dict") or score_state.get("model")
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ctrl_state = torch.load(ctrl_ckpt, map_location="cpu", weights_only=False)
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

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=50,
                              sigma_min=0.01, sigma_max=1.0)

    ds = datasets.FashionMNIST(
        root=str(ROOT / "data" / "tensors"), train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    feature_extractor = _build_feature_extractor(device)

    records = {"dsm_loss": [], "control_energy": [], "path_kl": [], "quality_loss": []}

    print(f"Running {n_batches} batches...")
    for i, (x_real, _) in enumerate(loader):
        if i >= n_batches:
            break
        x_real = x_real.to(device)

        sigma_val = 0.5
        sigma = torch.full((x_real.shape[0], 1, 1, 1), sigma_val, device=device)
        noise = torch.randn_like(x_real)
        noisy = x_real + sigma * noise
        target = -noise / sigma
        with torch.no_grad():
            score = model(noisy, sigma)
        dsm_loss = torch.mean((score - target) ** 2).item()

        with torch.no_grad():
            gen_x, control_energy, path_kl, _ = simulate_path(
                model, ctrl_policy, sde_cfg,
                batch_size=batch_size, device=device,
                channels=x_real.shape[1], height=x_real.shape[2], width=x_real.shape[3],
            )

        quality_loss = 0.0
        if feature_extractor is not None:
            try:
                ql = _feature_matching_loss(gen_x.detach(), x_real, feature_extractor)
                quality_loss = ql.item()
            except Exception:
                pass

        records["dsm_loss"].append(dsm_loss)
        records["control_energy"].append(control_energy.item())
        records["path_kl"].append(path_kl.item())
        records["quality_loss"].append(quality_loss)

        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{n_batches}: dsm={dsm_loss:.4f} ce={control_energy.item():.2e} "
                  f"pkL={path_kl.item():.4f} ql={quality_loss:.4f}")

    import statistics
    result = {}
    for key, vals in records.items():
        result[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n_batches": len(vals),
        }

    print("\nLoss term statistics at epoch 50 (Session 9 ConvControl):")
    for key, stats in result.items():
        print(f"  {key}: mean={stats['mean']:.6f}  std={stats['std']:.6f}  "
              f"min={stats['min']:.6f}  max={stats['max']:.6f}")

    out_path = ROOT / "data" / "results" / "loss_term_magnitudes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
