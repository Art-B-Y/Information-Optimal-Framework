"""Extended baseline analysis (Step 7C, Session 10).

Three analyses:
  A) NFE=25 added to baseline_evals_fashionmnist.json (DDPM/DDIM/SDE).
  B) Entropy production comparison: ITS (controlled) vs SDE (uncontrolled) vs DDPM.
     ITS has non-zero irreversible work; DDPM/SDE have zero control signal.
  C) Mode coverage comparison: class distribution entropy for DDPM vs DDIM vs SDE.

Results saved to data/results/extended_baseline_evals.json.

Usage:
    python scripts/extended_baseline_analysis.py \\
        --score-ckpt checkpoints/score_fmnist_v2/score_best.pt
    # With ITS entropy production (needs ctrl checkpoint):
    python scripts/extended_baseline_analysis.py \\
        --score-ckpt checkpoints/score_fmnist_v2/score_best.pt \\
        --ctrl-ckpt checkpoints/controlled_config_d_seed42/controlled_last.pt
    # Quick smoke test:
    python scripts/extended_baseline_analysis.py \\
        --score-ckpt checkpoints/score_fmnist_v2/score_best.pt --quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ---------------------------------------------------------------------------
# A: NFE=25 baseline eval
# ---------------------------------------------------------------------------

def run_nfe25_evals(
    score_model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    seed: int,
) -> list[dict]:
    """Evaluate DDPM/DDIM/SDE at NFE=25 and return result dicts."""
    from its.sde import ScoreSDEConfig
    from its.eval import EvaluationConfig
    from its.eval.evaluator import evaluate_sampler, evaluate_ddpm_baseline

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=25,
                              sigma_min=0.01, sigma_max=1.0)
    eval_cfg = EvaluationConfig(
        dataset_name="fashionmnist",
        num_samples=num_samples,
        batch_size=64,
        device=str(device),
        seed=seed,
    )
    results = []
    for sampler in ["ddpm", "ddim", "sde"]:
        print(f"  NFE=25 {sampler}...", end=" ", flush=True)
        try:
            if sampler == "sde":
                metrics = evaluate_sampler(score_model, None, sde_cfg, eval_cfg)
            else:
                metrics = evaluate_ddpm_baseline(score_model, sde_cfg, eval_cfg, baseline=sampler)
            rec = {"sampler": sampler, "nfe": 25, **metrics}
            print(f"FID={metrics['fid']:.2f}")
        except Exception as e:
            print(f"FAILED: {e}")
            rec = {"sampler": sampler, "nfe": 25, "error": str(e)}
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# B: Entropy production comparison
# ---------------------------------------------------------------------------

def _compute_entropy_production_its(
    score_model: torch.nn.Module,
    ctrl_policy: torch.nn.Module,
    device: torch.device,
    n_trajectories: int = 256,
    num_steps: int = 50,
) -> float:
    """Total entropy production proxy for controlled ITS: E[sum_t ||u_t||^2 * dt]."""
    beta_min, beta_max = 0.1, 10.0
    T = num_steps
    dt = 1.0 / T
    flat_dim = 784

    x = torch.randn(n_trajectories, 1, 28, 28, device=device)
    total_ep = 0.0
    with torch.no_grad():
        for step_idx in range(T):
            t_frac = 1.0 - step_idx / T
            sigma = (beta_min + (beta_max - beta_min) * t_frac) ** 0.5
            sigma_t = torch.tensor([sigma], device=device)
            score = score_model(x, sigma_t.expand(x.shape[0]))

            # Control signal
            if hasattr(ctrl_policy, "config"):
                # ConvControlPolicy: (B, C, H, W) input
                ctrl = ctrl_policy(x, torch.full((n_trajectories,), t_frac, device=device))
            else:
                x_flat = x.reshape(n_trajectories, flat_dim)
                t_tensor = torch.full((n_trajectories,), t_frac, device=device)
                ctrl = ctrl_policy(x_flat, t_tensor)

            ctrl_ep = float(ctrl.reshape(n_trajectories, -1).pow(2).sum(dim=1).mean().item()) * dt
            total_ep += ctrl_ep

            # SDE step
            beta_t = beta_min + (beta_max - beta_min) * t_frac
            drift = 0.5 * beta_t * (-x - score)
            noise = torch.randn_like(x) * (beta_t * dt) ** 0.5
            x = x + drift * dt + noise

    return total_ep


def compute_entropy_production_comparison(
    score_model: torch.nn.Module,
    ctrl_policy: Optional[torch.nn.Module],
    device: torch.device,
    n_trajectories: int = 256,
    num_steps: int = 50,
) -> dict:
    """
    Entropy production comparison table.
    DDPM/DDIM: no control signal → EP = 0 (deterministic schedule, u=0).
    SDE (uncontrolled): no control → EP = 0.
    ITS (controlled): non-zero u → EP > 0 (irreversible work done by controller).
    """
    print("  Computing entropy production comparison...")
    result: dict[str, float | str] = {
        "ddpm_entropy_production": 0.0,
        "ddim_entropy_production": 0.0,
        "sde_uncontrolled_entropy_production": 0.0,
    }
    if ctrl_policy is not None:
        print("  Computing ITS entropy production...", end=" ", flush=True)
        ep = _compute_entropy_production_its(score_model, ctrl_policy, device,
                                              n_trajectories, num_steps)
        result["its_entropy_production"] = ep
        print(f"EP={ep:.4f}")
    else:
        result["its_entropy_production"] = None
        result["note"] = "Pass --ctrl-ckpt to compute ITS entropy production"
    result["n_trajectories"] = n_trajectories
    result["num_steps"] = num_steps
    result["interpretation"] = (
        "DDPM/DDIM/SDE have EP=0 (no control perturbation). "
        "ITS EP > 0 represents minimum irreversible work needed to steer trajectories. "
        "Thermodynamically, this is the entropy production from the non-equilibrium control."
    )
    return result


# ---------------------------------------------------------------------------
# C: Mode coverage comparison (DDPM vs DDIM vs SDE at NFE=100)
# ---------------------------------------------------------------------------

class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_or_train_classifier(device: torch.device) -> _TinyClassifier:
    ckpt = ROOT / "checkpoints" / "fmnist_classifier.pt"
    model = _TinyClassifier().to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print("  Loaded cached TinyClassifier.")
        return model
    print("  Training TinyClassifier...")
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.FashionMNIST(ROOT / "data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
        print(f"    Classifier epoch {epoch+1}/5 loss={total_loss/len(loader):.4f}")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    return model


def _generate_and_classify(
    sampler_fn,
    shape: tuple,
    device: torch.device,
    classifier: _TinyClassifier,
    num_samples: int,
    batch_size: int = 64,
) -> dict:
    classifier.eval()
    generated = []
    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            result = sampler_fn(bs)
            sample = result[0] if isinstance(result, tuple) else result
            generated.append(sample.clamp(-1, 1).cpu())
            remaining -= bs
    gen_tensor = torch.cat(generated, dim=0)[:num_samples]

    all_labels = []
    with torch.no_grad():
        for i in range(0, len(gen_tensor), batch_size):
            batch = gen_tensor[i:i+batch_size].to(device)
            logits = classifier(batch)
            all_labels.extend(logits.argmax(dim=1).cpu().tolist())

    n = len(all_labels)
    counts = [all_labels.count(c) for c in range(10)]
    fractions = [c / n for c in counts]
    entropy = -sum(p * math.log(p + 1e-12) for p in fractions if p > 0)
    return {
        "counts": counts,
        "fractions": fractions,
        "class_names": CLASS_NAMES,
        "entropy": entropy,
        "max_entropy": math.log(10),
        "entropy_fraction": entropy / math.log(10),
        "n_samples": n,
    }


def compute_mode_coverage_comparison(
    score_model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    nfe: int = 100,
) -> dict:
    from its.sde import ScoreSDEConfig
    from its.samplers.ddpm import DDPMSampler, DDPMSamplerConfig
    from its.sde.score_sde import ScoreSDESimulator

    classifier = _load_or_train_classifier(device)
    shape = (1, 28, 28)
    results = {}

    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=10.0, num_steps=nfe,
                              sigma_min=0.01, sigma_max=1.0)

    for sampler_name in ["sde", "ddpm", "ddim"]:
        print(f"  Mode coverage {sampler_name} NFE={nfe}...", end=" ", flush=True)
        try:
            if sampler_name == "sde":
                simulator = ScoreSDESimulator(score_model, sde_cfg)
                fn = lambda bs: simulator.sample((bs,) + shape, device=device)
            elif sampler_name == "ddpm":
                cfg = DDPMSamplerConfig(num_steps=nfe, use_ddim=False)
                ddpm = DDPMSampler(score_model, cfg)
                fn = lambda bs: ddpm.sample((bs,) + shape, device=device)
            else:  # ddim
                cfg = DDPMSamplerConfig(num_steps=nfe, use_ddim=True, ddim_eta=0.0)
                ddim = DDPMSampler(score_model, cfg)
                fn = lambda bs: ddim.sample((bs,) + shape, device=device)

            cov = _generate_and_classify(fn, shape, device, classifier, num_samples)
            print(f"entropy={cov['entropy']:.3f} ({cov['entropy_fraction']*100:.1f}% of max)")
            results[sampler_name] = cov
        except Exception as e:
            print(f"FAILED: {e}")
            results[sampler_name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-ckpt", required=True, help="Path to score model checkpoint")
    parser.add_argument("--ctrl-ckpt", default=None,
                        help="Path to ITS controller checkpoint (for entropy production)")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Samples per sampler (mode coverage + NFE=25)")
    parser.add_argument("--nfe-coverage", type=int, default=100,
                        help="NFE for mode coverage analysis")
    parser.add_argument("--entropy-trajectories", type=int, default=256,
                        help="Trajectories for entropy production estimate")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced N=128 smoke test (skips full analysis)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-nfe25", action="store_true",
                        help="Skip NFE=25 baseline evals (already done)")
    parser.add_argument("--skip-coverage", action="store_true",
                        help="Skip mode coverage analysis")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_samples = 128 if args.quick else args.num_samples
    n_trajectories = 32 if args.quick else args.entropy_trajectories

    print(f"=== Extended Baseline Analysis (Step 7C) ===")
    print(f"Device: {device}, N={num_samples}, seed={args.seed}")

    from its.models import ScoreUNetConfig, build_score_model

    # Load score model
    state = torch.load(args.score_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = state.get("score_config") or state.get("model_config")
    if cfg_dict:
        model_cfg = ScoreUNetConfig(**cfg_dict)
    else:
        model_cfg = ScoreUNetConfig(in_channels=1, base_channels=64, channel_mults=(1, 2, 2),
                                    use_time_embedding=True, time_embed_dim=128)
    score_model = build_score_model(model_cfg).to(device)
    score_sd = state.get("score_state_dict") or state.get("model_state_dict") or state
    score_model.load_state_dict(score_sd, strict=True)
    score_model.eval()
    print(f"Loaded score model from {args.score_ckpt}")

    # Load controller if provided
    ctrl_policy = None
    if args.ctrl_ckpt:
        ctrl_state = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
        ctrl_cfg_d = ctrl_state.get("control_config", {})
        ctrl_type = ctrl_cfg_d.get("type", "conv" if "conv" in args.ctrl_ckpt.lower() else "mlp")

        if "base_channels" in ctrl_cfg_d:
            from its.controllers.neural_control import ConvControlConfig, build_conv_control_policy
            conv_cfg = ConvControlConfig(
                in_channels=ctrl_cfg_d.get("in_channels", 1),
                base_channels=ctrl_cfg_d.get("base_channels", 64),
                num_blocks=ctrl_cfg_d.get("num_blocks", 3),
                time_embed_dim=ctrl_cfg_d.get("time_embed_dim", 128),
                dropout=ctrl_cfg_d.get("dropout", 0.0),
            )
            ctrl_policy = build_conv_control_policy(conv_cfg, device=device)
        else:
            from its.controllers.neural_control import ControlConfig, build_control_policy
            ctrl_cfg = ControlConfig(state_dim=ctrl_cfg_d.get("state_dim", 784),
                                      hidden_dim=ctrl_cfg_d.get("hidden_dim", 512),
                                      output_dim=ctrl_cfg_d.get("output_dim", 784))
            ctrl_policy = build_control_policy(ctrl_cfg, device=device)

        ctrl_key = ctrl_state.get("control_state_dict") or ctrl_state.get("control")
        ctrl_policy.load_state_dict(ctrl_key, strict=True)
        ctrl_policy.eval()
        print(f"Loaded controller from {args.ctrl_ckpt}")

    results: dict = {
        "score_ckpt": args.score_ckpt,
        "ctrl_ckpt": args.ctrl_ckpt,
        "num_samples": num_samples,
        "seed": args.seed,
    }

    # A: NFE=25 evals
    if not args.skip_nfe25:
        print("\n--- Part A: NFE=25 baseline evals ---")
        nfe25_results = run_nfe25_evals(score_model, device, num_samples, args.seed)
        results["nfe25_evals"] = nfe25_results

        # Merge with existing baseline_evals if present
        existing_path = ROOT / "data" / "results" / "baseline_evals_fashionmnist.json"
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text())
                # Deduplicate: remove any existing NFE=25 entries then add new ones
                existing_runs = [r for r in existing.get("runs", []) if r.get("nfe") != 25]
                existing_runs.extend(nfe25_results)
                existing["runs"] = existing_runs
                existing_path.write_text(json.dumps(existing, indent=2))
                print(f"  Updated {existing_path} with NFE=25 results.")
            except Exception as e:
                print(f"  Could not update existing baseline_evals: {e}")

    # B: Entropy production comparison
    print("\n--- Part B: Entropy production comparison ---")
    ep_result = compute_entropy_production_comparison(
        score_model, ctrl_policy, device, n_trajectories, num_steps=50
    )
    results["entropy_production"] = ep_result

    # C: Mode coverage
    if not args.skip_coverage:
        print("\n--- Part C: Mode coverage comparison ---")
        coverage_result = compute_mode_coverage_comparison(
            score_model, device, num_samples, nfe=args.nfe_coverage
        )
        results["mode_coverage"] = coverage_result

        print("\nMode coverage summary (entropy fraction of max log(10)):")
        for name, cov in coverage_result.items():
            if "error" not in cov:
                print(f"  {name:6s}: {cov['entropy_fraction']*100:.1f}% of max")

    # Save results
    out_path = ROOT / "data" / "results" / "extended_baseline_evals.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nExtended baseline analysis saved -> {out_path}")

    # Summary
    print("\n=== Summary ===")
    if "nfe25_evals" in results:
        for r in results["nfe25_evals"]:
            if "error" not in r:
                print(f"  {r['sampler']:6s} NFE=25: FID={r.get('fid', '?'):.2f}")
    ep = results.get("entropy_production", {})
    print(f"  Entropy production: DDPM=0.0, SDE=0.0, ITS={ep.get('its_entropy_production', 'N/A')}")


if __name__ == "__main__":
    main()
