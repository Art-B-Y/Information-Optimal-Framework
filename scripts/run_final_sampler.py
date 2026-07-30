from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import List, Optional

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.utils import save_image

from its.controllers import build_control_policy
from its.data import get_dataset_stats
from its.models import ScoreUNetConfig, build_score_model
from its.utils import clamp_unit, denormalize


def parse_overrides(raw: Optional[str], repeated: Optional[List[str]]) -> List[str]:
    overrides: List[str] = []
    if repeated:
        overrides.extend(repeated)
    if raw:
        overrides.extend(raw.split())
    return overrides


def _resolve_checkpoint(
    explicit: Optional[Path],
    training_cfg,
    root: Path,
) -> Optional[Path]:
    if explicit is not None:
        return explicit
    ckpt_dir = getattr(training_cfg, "checkpoint_dir", None)
    if not ckpt_dir:
        return None
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.exists():
        return None
    candidates = sorted(ckpt_path.glob("score_epoch_*.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from the final-phase ITS controlled SDE.")
    parser.add_argument("--config-name", "-c", default="final_phase", help="Hydra config to use.")
    parser.add_argument("--override", "-o", action="append", default=[], help="Hydra override (repeatable).")
    parser.add_argument("--overrides", default=None, help="Space-separated Hydra overrides.")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Number of samples.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--output", type=str, default="data/results/final_samples.pt", help="Where to save samples.")
    parser.add_argument("--grid-output", type=str, default=None, help="Optional PNG grid output path.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    hydra_overrides = parse_overrides(args.overrides, args.override)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="its-sample"):
        cfg: DictConfig = compose(config_name=args.config_name, overrides=hydra_overrides)

    bundle = instantiate(cfg.experiment)
    sampler = bundle["sampler"]
    score_model = bundle["model"]
    training_cfg = bundle.get("score_training")
    dataset_cfg = bundle.get("dataset_cfg")
    device = next(score_model.parameters()).device

    ckpt_path = _resolve_checkpoint(Path(args.checkpoint) if args.checkpoint else None, training_cfg, root)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # If checkpoint contains saved model architecture config, rebuild the
        # model from it to avoid silent shape mismatches (Bug 1 fix).
        if "model_config" in state:
            saved_cfg = ScoreUNetConfig(**state["model_config"])
            # Verify the saved config matches what was instantiated from Hydra.
            instantiated_cfg = getattr(score_model, "config", None)
            if instantiated_cfg is not None:
                inst_dict = dataclasses.asdict(instantiated_cfg)
                saved_dict = dataclasses.asdict(saved_cfg)
                if inst_dict != saved_dict:
                    raise ValueError(
                        f"Checkpoint architecture config does not match instantiated model.\n"
                        f"  Checkpoint: {saved_dict}\n"
                        f"  Instantiated: {inst_dict}\n"
                        f"Use a Hydra override to match the checkpoint's architecture "
                        f"(e.g. experiment.model.base_channels={saved_cfg.base_channels})."
                    )
            # Rebuild model from checkpoint config so load is always safe.
            score_model = build_score_model(saved_cfg)
        state_dict = state.get("model", state)
        score_model.load_state_dict(state_dict)
        score_model.to(device)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Sampling with initialised weights.")

    sampler.model = score_model

    control_policy = None
    if cfg.get("use_control", False):
        control_policy = build_control_policy(bundle["control_cfg"], device=device)

    samples = sampler.sample((args.batch, score_model.config.in_channels, 32, 32), device=device, control=control_policy)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_cpu = samples.cpu()
    torch.save(samples_cpu, output_path)
    print(f"Saved samples to {output_path}")

    if args.grid_output:
        if dataset_cfg is not None and getattr(dataset_cfg, "normalize", True):
            mean, std = get_dataset_stats(dataset_cfg.name)
            samples_vis = clamp_unit(denormalize(samples_cpu, mean, std))
        else:
            samples_vis = clamp_unit(samples_cpu)
        grid_output = Path(args.grid_output)
        grid_output.parent.mkdir(parents=True, exist_ok=True)
        nrow = int(math.sqrt(samples_vis.shape[0])) or 1
        save_image(samples_vis, grid_output, nrow=nrow)
        print(f"Saved PNG grid to {grid_output}")


if __name__ == "__main__":
    main()
