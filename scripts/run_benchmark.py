from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from its.eval import EvaluationConfig, evaluate_sampler
from its.eval.evaluator import evaluate_ddpm_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ITS against baseline samplers.")
    parser.add_argument("--config-name", "-c", default="controlled_mnist", help="Hydra config to use.")
    parser.add_argument("--override", "-o", action="append", default=[], help="Hydra override (repeatable).")
    parser.add_argument("--overrides", default=None, help="Space-separated Hydra overrides.")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of generated samples for evaluation.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for sampling and metrics.")
    parser.add_argument("--device", type=str, default=None, help="Device override for evaluation.")
    parser.add_argument("--output", type=str, default="summaries/benchmark_results.json", help="Where to save results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    hydra_overrides: List[str] = []
    if args.override:
        hydra_overrides.extend(args.override)
    if args.overrides:
        hydra_overrides.extend(args.overrides.split())

    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="its-benchmark"):
        cfg: DictConfig = compose(config_name=args.config_name, overrides=hydra_overrides)

    bundle = instantiate(cfg.experiment)
    score_model = bundle["model"]
    control_policy = bundle["control"]
    training_cfg = bundle["training"]
    sde_cfg = bundle["sde"]

    device = args.device or getattr(training_cfg, "device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    score_model.to(device)
    control_policy.to(device)

    eval_cfg = EvaluationConfig(
        dataset_name=getattr(training_cfg, "dataset_name", "fashionmnist"),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
    )

    results = {}
    results["controlled"] = evaluate_sampler(score_model, control_policy, sde_cfg, eval_cfg)
    results["baseline_sde"] = evaluate_sampler(score_model, None, sde_cfg, eval_cfg)
    results["ddpm"] = evaluate_ddpm_baseline(score_model, sde_cfg, eval_cfg, baseline="ddpm")
    results["ddim"] = evaluate_ddpm_baseline(score_model, sde_cfg, eval_cfg, baseline="ddim")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
