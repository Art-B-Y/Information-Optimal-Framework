"""Gradient flow diagnostic for ConvControl collapse analysis (Session 10 Step 1B).

Loads the Session 9 ConvControl checkpoint (epoch 50), runs one forward pass
with a 16-sample batch, and measures per-loss-term gradient contributions
on the controller's output layer and first conv layer.

Key metrics computed:
  - grad_mag_output_layer: mean |grad| on output_proj weights
  - grad_mag_first_conv: mean |grad| on blocks[0].conv1 weights
  - quality_vs_control_energy_ratio: quality grad / control_energy grad on output layer
  - path_kl_vs_control_energy_ratio: path_kl grad / control_energy grad on output layer

If control_energy grad > quality grad by >10x: control energy dominates → detach it.
If path_kl grad >= quality grad: path_kl also suppressing controller.

Usage:
    python scripts/diagnose_gradient_flow.py
    python scripts/diagnose_gradient_flow.py --batch-size 16 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _get_grad_mag(module: nn.Module) -> float:
    """Mean absolute gradient magnitude over all parameters of a module."""
    grads = []
    for p in module.parameters():
        if p.grad is not None:
            grads.append(p.grad.abs().mean().item())
    return sum(grads) / len(grads) if grads else 0.0


def _zero_grads(policy: nn.Module) -> None:
    for p in policy.parameters():
        if p.grad is not None:
            p.grad.zero_()


def diagnose(
    score_ckpt: Path,
    ctrl_ckpt: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy, ConvControlPolicy
    from its.sde import ScoreSDEConfig
    from its.training.controlled_score_training import simulate_path, _build_feature_extractor, _feature_matching_loss
    from its.physics.fluctuation import jarzynski_work_estimate
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Load score model
    score_state = torch.load(score_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                 use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(device)
    sd = score_state.get("score_state_dict") or score_state.get("model_state_dict") or score_state.get("model")
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load ConvControl checkpoint
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
    ctrl_policy.train()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=50,
                              sigma_min=0.01, sigma_max=1.0)

    # Load a small FashionMNIST batch
    ds = datasets.FashionMNIST(
        root=str(ROOT / "data" / "tensors"),
        train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    x_real, _ = next(iter(loader))
    x_real = x_real[:batch_size].to(device)

    # Build feature extractor for quality loss
    feature_extractor = _build_feature_extractor(device)

    # Optimizer (just for gradient tracking, we don't step)
    optim = torch.optim.AdamW(ctrl_policy.parameters(), lr=5e-4)

    # Compute DSM loss (standard score matching on the real batch)
    sigma_val = 0.5
    sigma = torch.full((x_real.shape[0], 1, 1, 1), sigma_val, device=device)
    noise = torch.randn_like(x_real)
    noisy = x_real + sigma * noise
    target = -noise / sigma
    with torch.no_grad():
        score = model(noisy, sigma)
    dsm_loss = torch.mean((score - target) ** 2)

    # Run controlled SDE path for control_energy, path_kl, and generated samples
    gen_x, control_energy, path_kl, jarzynski = simulate_path(
        model, ctrl_policy, sde_cfg,
        batch_size=batch_size, device=device,
        channels=x_real.shape[1], height=x_real.shape[2], width=x_real.shape[3],
    )

    # Feature matching quality loss (original)
    quality_loss = torch.tensor(0.0, device=device)
    if feature_extractor is not None:
        try:
            quality_loss = _feature_matching_loss(gen_x, x_real, feature_extractor)
        except Exception as e:
            print(f"[WARN] quality_loss failed: {e}")

    print(f"\nLoss magnitudes:")
    print(f"  DSM loss:        {dsm_loss.item():.4f}")
    print(f"  Control energy:  {control_energy.item():.6f}")
    print(f"  Path KL:         {path_kl.item():.6f}")
    print(f"  Quality loss:    {quality_loss.item():.4f}")

    # --- Measure per-term gradient on controller output layer ---
    output_layer = ctrl_policy.output_proj
    first_conv = ctrl_policy.blocks[0].conv1

    results = {}

    def _measure_grads(loss_tensor: torch.Tensor, label: str) -> dict:
        optim.zero_grad()
        if loss_tensor.requires_grad or loss_tensor.grad_fn is not None:
            loss_tensor.backward(retain_graph=True)
        out_mag = _get_grad_mag(output_layer)
        first_mag = _get_grad_mag(first_conv)
        print(f"  {label}: output_grad={out_mag:.3e}  first_conv_grad={first_mag:.3e}")
        return {"output_layer": out_mag, "first_conv": first_mag}

    print(f"\nPer-term gradient magnitudes on controller:")

    # Individual loss terms — measure gradient of each term alone
    # control_weight = 1e-2 (matching Session 9 training config)
    control_weight = 0.01
    path_kl_weight = 0.1
    quality_weight = 0.01

    r_dsm = _measure_grads(dsm_loss, "DSM loss")
    r_ce = _measure_grads(control_weight * control_energy, "Control energy (weighted)")
    r_pkL = _measure_grads(path_kl_weight * path_kl, "Path KL (weighted)")

    if quality_loss.item() > 0:
        r_ql = _measure_grads(quality_weight * quality_loss, "Quality loss (weighted)")
    else:
        r_ql = {"output_layer": 0.0, "first_conv": 0.0}
        print("  Quality loss: skipped (zero)")

    # Now compute the combined loss gradient
    optim.zero_grad()
    total_loss = (dsm_loss
                  + control_weight * control_energy
                  + path_kl_weight * path_kl
                  + quality_weight * quality_loss)
    total_loss.backward()
    total_out_mag = _get_grad_mag(output_layer)
    total_first_mag = _get_grad_mag(first_conv)

    # Compute ratios
    eps = 1e-15
    q_vs_ce = r_ql["output_layer"] / (r_ce["output_layer"] + eps)
    pkL_vs_ce = r_pkL["output_layer"] / (r_ce["output_layer"] + eps)
    q_vs_pkL = r_ql["output_layer"] / (r_pkL["output_layer"] + eps)
    ce_vs_q = r_ce["output_layer"] / (r_ql["output_layer"] + eps)

    print(f"\nGradient dominance analysis (output layer):")
    print(f"  Control energy grad:  {r_ce['output_layer']:.3e}")
    print(f"  Path KL grad:         {r_pkL['output_layer']:.3e}")
    print(f"  Quality grad:         {r_ql['output_layer']:.3e}")
    print(f"  DSM grad:             {r_dsm['output_layer']:.3e}")
    print(f"  Total grad:           {total_out_mag:.3e}")
    print(f"")
    print(f"  CE / quality ratio:   {ce_vs_q:.1f}x  {'<-- DOMINANT' if ce_vs_q > 10 else ''}")
    print(f"  PathKL / quality:     {pkL_vs_ce:.1f}x  {'<-- SUPPRESSING' if pkL_vs_ce > 1 else ''}")
    print(f"  Quality / CE:         {q_vs_ce:.3f}  {'<-- quality too weak' if q_vs_ce < 0.1 else '<-- OK'}")

    quality_grad = r_ql["output_layer"]
    pkL_grad = r_pkL["output_layer"]
    ce_grad = r_ce["output_layer"]

    if quality_grad == 0.0 and pkL_grad > 0:
        diagnosis = "PATH_KL_DOMINANT_QUALITY_BROKEN"
        print(f"\n[DIAGNOSIS] Quality gradient is ZERO (broken gradient chain through SDE+Inception).")
        print(f"  Path KL gradient ({pkL_grad:.3e}) is the sole driver of controller collapse.")
        print("  --> Must: (1) detach path_kl or set path_kl_weight=0, "
              "(2) use REINFORCE for quality signal.")
    elif ce_vs_q > 10:
        diagnosis = "CONTROL_ENERGY_DOMINANT"
        print(f"\n[DIAGNOSIS] Control energy gradient dominates quality by {ce_vs_q:.0f}x.")
        print("  --> Must detach control_energy from controller gradient (Redesign A).")
    elif pkL_grad > quality_grad:
        diagnosis = "PATH_KL_DOMINANT"
        print(f"\n[DIAGNOSIS] Path KL gradient ({pkL_grad:.3e}) "
              f"> quality gradient ({quality_grad:.3e}).")
        print("  --> Reduce path_kl_weight or also detach path_kl (or both).")
    else:
        diagnosis = "QUALITY_SIGNAL_TOO_WEAK"
        print(f"\n[DIAGNOSIS] Quality signal is weak but not dominated by other terms.")
        print("  --> Increase quality_weight and use per-sample trajectory quality loss.")

    result = {
        "loss_magnitudes": {
            "dsm_loss": dsm_loss.item(),
            "control_energy": control_energy.item(),
            "path_kl": path_kl.item(),
            "quality_loss": quality_loss.item(),
        },
        "weighted_loss_magnitudes": {
            "dsm_loss": dsm_loss.item(),
            "control_energy_weighted": control_weight * control_energy.item(),
            "path_kl_weighted": path_kl_weight * path_kl.item(),
            "quality_loss_weighted": quality_weight * quality_loss.item(),
        },
        "gradient_magnitudes_output_layer": {
            "dsm_loss": r_dsm["output_layer"],
            "control_energy_weighted": r_ce["output_layer"],
            "path_kl_weighted": r_pkL["output_layer"],
            "quality_loss_weighted": r_ql["output_layer"],
            "total": total_out_mag,
        },
        "gradient_magnitudes_first_conv": {
            "dsm_loss": r_dsm["first_conv"],
            "control_energy_weighted": r_ce["first_conv"],
            "path_kl_weighted": r_pkL["first_conv"],
            "quality_loss_weighted": r_ql["first_conv"],
        },
        "dominance_ratios": {
            "control_energy_vs_quality": ce_vs_q,
            "path_kl_vs_quality": pkL_vs_ce,
            "quality_vs_control_energy": q_vs_ce,
        },
        "diagnosis": diagnosis,
        "redesign_implications": {
            "detach_control_energy": ce_vs_q > 10 or ce_vs_q > 1,
            "reduce_path_kl_weight": r_pkL["output_layer"] > r_ql["output_layer"],
            "increase_quality_weight": r_ql["output_layer"] < 1e-6,
            "use_trajectory_quality": True,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt",
                        default=str(ROOT / "checkpoints" / "score_fmnist_v2" / "score_best.pt"))
    parser.add_argument("--ctrl-ckpt",
                        default=str(ROOT / "checkpoints" / "controlled_conv_seed42" / "controlled_last.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out",
                        default=str(ROOT / "data" / "results" / "gradient_flow_diagnosis.json"))
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Score ckpt: {args.score_ckpt}")
    print(f"Ctrl ckpt:  {args.ctrl_ckpt}")

    result = diagnose(
        Path(args.score_ckpt), Path(args.ctrl_ckpt),
        device, args.batch_size,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
