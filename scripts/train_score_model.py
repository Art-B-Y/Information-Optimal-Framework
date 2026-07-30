from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from its.training import train_score_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train score-based models for the ITS final phase.")
    parser.add_argument("--config-name", "-c", default="final_phase", help="Hydra config for training.")
    parser.add_argument("--override", "-o", action="append", default=[], help="Hydra override (repeatable).")
    parser.add_argument("--overrides", default=None, help="Space-separated Hydra overrides.")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--base-channels", type=int, default=None, help="Override UNet base channels.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataset batch size.")
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
    if args.resume_from:
        hydra_overrides.append(f"experiment.score_training.resume_from={args.resume_from}")
    if args.epochs is not None:
        hydra_overrides.append(f"experiment.score_training.epochs={args.epochs}")
    if args.base_channels is not None:
        hydra_overrides.append(f"experiment.model.base_channels={args.base_channels}")
    if args.batch_size is not None:
        hydra_overrides.append(f"experiment.dataset.batch_size={args.batch_size}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="its-train"):
        cfg: DictConfig = compose(config_name=args.config_name, overrides=hydra_overrides)

    bundle = instantiate(cfg.experiment)
    training_cfg = bundle["score_training"]
    print("Starting score model training...")
    metrics = train_score_model(training_cfg)
    print(f"Training finished with metrics: {metrics}")


if __name__ == "__main__":
    main()
