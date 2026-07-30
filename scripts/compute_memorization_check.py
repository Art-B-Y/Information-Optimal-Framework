"""Compute nearest-neighbor memorization check (Step 4D, Session 7).

Compares generated samples to training set in pixel space.
Fraction below threshold signals potential memorization.

Usage:
    python scripts/compute_memorization_check.py --score-ckpt checkpoints/score_fmnist_v2/score_best.pt
    python scripts/compute_memorization_check.py --score-ckpt <ckpt> --num-generated 1000 --threshold 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True)
    parser.add_argument("--num-generated", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="NN distance threshold for memorization flag")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from its.models import ScoreUNetConfig, build_score_model
    from its.sde import ScoreSDEConfig
    from its.sde.score_sde import ScoreSDESimulator
    from torchvision import datasets, transforms

    # Load score model
    state = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                 use_time_embedding=True, time_embed_dim=128)
    model = build_score_model(model_cfg).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=args.nfe,
                              sigma_min=0.01, sigma_max=1.0)
    simulator = ScoreSDESimulator(model, sde_cfg)

    # Generate samples
    print(f"Generating {args.num_generated} samples...")
    generated_flat = []
    remaining = args.num_generated
    with torch.no_grad():
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            result = simulator.sample((bs, 1, 28, 28), device=device)
            samples = result[0] if isinstance(result, tuple) else result
            flat = samples.clamp(-1, 1).reshape(bs, -1).cpu()
            generated_flat.append(flat)
            remaining -= bs
    generated = torch.cat(generated_flat, dim=0)[:args.num_generated]

    # Load training set
    print("Loading training set for NN comparison...")
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(ROOT / "data", train=True, transform=tf, download=True)
    train_flat = torch.stack([train_ds[i][0].reshape(-1) for i in range(len(train_ds))])

    # Nearest-neighbor distances (batched to avoid OOM)
    print("Computing nearest-neighbor distances...")
    nn_dists = []
    for g in generated:
        dists = (train_flat - g.unsqueeze(0)).norm(dim=1)
        nn_dists.append(float(dists.min().item()))

    fraction_below = sum(1 for d in nn_dists if d < args.threshold) / len(nn_dists)
    nn_mean = sum(nn_dists) / len(nn_dists)
    nn_min = min(nn_dists)
    nn_max = max(nn_dists)

    result = {
        "num_generated": len(nn_dists),
        "num_train": len(train_flat),
        "threshold": args.threshold,
        "nn_distances_mean": nn_mean,
        "nn_distances_min": nn_min,
        "nn_distances_max": nn_max,
        "fraction_below_threshold": fraction_below,
        "memorization_concern": fraction_below > 0.01,
        "score_ckpt": args.score_ckpt,
        "nfe": args.nfe,
    }

    out = ROOT / "data" / "results" / "memorization_check.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nMemorization check complete:")
    print(f"  NN mean distance: {nn_mean:.4f}")
    print(f"  Fraction below threshold ({args.threshold}): {fraction_below:.4f}")
    print(f"  Memorization concern: {result['memorization_concern']}")
    print(f"  Result -> {out}")


if __name__ == "__main__":
    main()
