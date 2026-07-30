from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from its.training import run_training

app = typer.Typer(help="Launch ITS beta-phase experiments.")


@app.command()
def run(
    config_name: str = typer.Option("config", help="Base Hydra config name."),
    overrides: List[str] = typer.Option([], help="Additional Hydra overrides."),
) -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="its-run"):
        cfg: DictConfig = compose(config_name=config_name, overrides=list(overrides))

    bundle = instantiate(cfg.experiment)
    sampler = bundle["sampler"]
    controller = bundle["controller"]
    training_cfg = bundle["training"]
    quality_fn = bundle.get("quality_fn")

    metrics = run_training(sampler, controller, training_cfg, quality_fn=quality_fn)
    typer.secho(f"Final metrics: {metrics}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
