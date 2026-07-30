from __future__ import annotations

from pathlib import Path
from typing import List

import argparse
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from its.training.controlled_score_training import train_controlled_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled score-based models (joint score + control).")
    parser.add_argument("--config-name", "-c", default="controlled_mnist", help="Hydra config to use.")
    parser.add_argument("--override", "-o", action="append", default=[], help="Hydra override (repeatable).")
    parser.add_argument("--overrides", default=None, help="Space-separated Hydra overrides.")
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

    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="its-train-controlled"):
        cfg: DictConfig = compose(config_name=args.config_name, overrides=hydra_overrides)

    bundle = instantiate(cfg.experiment)
    training_cfg = bundle["training"]
    metrics = train_controlled_score(training_cfg)
    print(f"Training finished with metrics: {metrics}")


if __name__ == "__main__":
    main()
