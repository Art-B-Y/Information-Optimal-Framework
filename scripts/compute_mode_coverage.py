"""Mode coverage analysis via class distribution (Step 4E, Session 7).

Trains a simple CNN classifier on FashionMNIST (if not cached), then
classifies generated samples from each sampler and reports per-class fractions.

Usage:
    python scripts/compute_mode_coverage.py --score-ckpt checkpoints/score_fmnist_v2/score_best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASSIFIER_CKPT = ROOT / "checkpoints" / "fmnist_classifier.pt"
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(device: torch.device) -> TinyClassifier:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("Training FashionMNIST classifier...")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.FashionMNIST(ROOT / "data", train=True, transform=tf, download=True)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    model = TinyClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"  Classifier epoch {epoch+1}/5 loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), CLASSIFIER_CKPT)
    return model


def classify_samples(samples: torch.Tensor, classifier: TinyClassifier,
                     device: torch.device, batch_size: int = 64) -> list[int]:
    labels = []
    classifier.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size].to(device)
            logits = classifier(batch)
            labels.extend(logits.argmax(dim=1).cpu().tolist())
    return labels


def compute_coverage(labels: list[int], n_classes: int = 10) -> dict:
    counts = [0] * n_classes
    for lbl in labels:
        counts[lbl] += 1
    n = len(labels)
    fractions = [c / n for c in counts]
    entropy = -sum(p * math.log(p + 1e-12) for p in fractions if p > 0)
    return {
        "counts": counts,
        "fractions": fractions,
        "class_names": CLASS_NAMES,
        "entropy": entropy,
        "max_entropy": math.log(n_classes),
        "entropy_fraction": entropy / math.log(n_classes),
        "n_samples": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load/train classifier
    classifier = TinyClassifier().to(device)
    if CLASSIFIER_CKPT.exists():
        classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=device,
                                               weights_only=True))
        print("Loaded cached classifier.")
    else:
        classifier = train_classifier(device)

    from its.models import ScoreUNetConfig, build_score_model
    from its.sde import ScoreSDEConfig
    from its.sde.score_sde import ScoreSDESimulator

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

    print(f"Generating {args.num_samples} samples...")
    generated = []
    remaining = args.num_samples
    with torch.no_grad():
        while remaining > 0:
            bs = min(64, remaining)
            result = simulator.sample((bs, 1, 28, 28), device=device)
            samples = result[0] if isinstance(result, tuple) else result
            generated.append(samples.clamp(-1, 1).cpu())
            remaining -= bs
    generated_tensor = torch.cat(generated, dim=0)[:args.num_samples]
    # Rescale from [-1,1] to [0,1] for classifier (trained on Normalize((0.5,),(0.5,)))
    # Classifier expects [-1,1] normalized input, so pass directly
    labels = classify_samples(generated_tensor, classifier, device)
    coverage = compute_coverage(labels)

    result = {"score_sde": coverage}

    out = ROOT / "data" / "results" / "mode_coverage.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nMode coverage:")
    for name, frac in zip(CLASS_NAMES, coverage["fractions"]):
        print(f"  {name}: {frac:.3f}")
    print(f"  Entropy: {coverage['entropy']:.3f} / max {coverage['max_entropy']:.3f} "
          f"({coverage['entropy_fraction']*100:.1f}%)")
    print(f"  Result -> {out}")


if __name__ == "__main__":
    main()
