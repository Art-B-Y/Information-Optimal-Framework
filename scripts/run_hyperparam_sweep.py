"""Hyperparameter sweep for controlled score-based training on FashionMNIST.

Runs 6 configurations (A-F) for 1 epoch each on a 2000-sample subset of
FashionMNIST with batch_size=32 and num_steps=50.  After each training run,
evaluates with 256 generated samples (--quick mode).

Results saved to: data/results/hyperparam_sweep_fmnist.json

Usage:
    python scripts/run_hyperparam_sweep.py
    python scripts/run_hyperparam_sweep.py --configs A B C   # subset
    python scripts/run_hyperparam_sweep.py --skip-eval       # training only
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make sure the src directory is importable when run from the project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch

from its.models import ScoreUNetConfig, build_score_model
from its.controllers import ControlConfig, build_control_policy
from its.sde import ScoreSDEConfig, ScoreSDESimulator
from its.eval import EvaluationConfig, evaluate_sampler
from its.training.controlled_score_training import (
    ControlledScoreTrainingConfig,
    train_controlled_score,
)


# ---------------------------------------------------------------------------
# Sweep configuration table
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    key: str
    control_weight: float
    path_kl_weight: float
    quality_weight: float
    description: str


SWEEP_CONFIGS: list[SweepPoint] = [
    SweepPoint("A", control_weight=0.001, path_kl_weight=0.0,  quality_weight=0.0,  description="Low ctrl, no path-KL, no quality"),
    SweepPoint("B", control_weight=0.01,  path_kl_weight=0.0,  quality_weight=0.0,  description="Med ctrl, no path-KL, no quality"),
    SweepPoint("C", control_weight=0.01,  path_kl_weight=0.1,  quality_weight=0.0,  description="Med ctrl, path-KL=0.1, no quality"),
    SweepPoint("D", control_weight=0.01,  path_kl_weight=0.1,  quality_weight=0.01, description="Med ctrl, path-KL=0.1, quality=0.01"),
    SweepPoint("E", control_weight=0.1,   path_kl_weight=0.1,  quality_weight=0.01, description="High ctrl, path-KL=0.1, quality=0.01"),
    SweepPoint("F", control_weight=0.01,  path_kl_weight=0.5,  quality_weight=0.01, description="Med ctrl, high path-KL=0.5, quality=0.01"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_training_config(
    pt: SweepPoint,
    checkpoint_dir: str,
    jsonl_log: str,
) -> ControlledScoreTrainingConfig:
    """Build ControlledScoreTrainingConfig for one sweep point."""
    return ControlledScoreTrainingConfig(
        epochs=1,
        lr=2e-4,
        sigma_min=0.01,
        sigma_max=1.0,
        grad_clip=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_interval=50,
        dataset_name="fashionmnist",
        batch_size=32,
        num_workers=0,
        dataset_subset_size=2000,
        model=ScoreUNetConfig(in_channels=1, base_channels=32),
        control=ControlConfig(
            state_dim=784,   # 28*28 FashionMNIST
            hidden_dim=128,
            depth=3,
            time_embedding_dim=8,
        ),
        sde=ScoreSDEConfig(
            beta_min=0.1,
            beta_max=5.0,
            num_steps=50,
            clamp=5.0,
            control_weight=1.0,
            sigma_min=0.01,
            sigma_max=1.0,
        ),
        control_weight=pt.control_weight,
        path_kl_weight=pt.path_kl_weight,
        quality_weight=pt.quality_weight,
        ema_decay=0.999,
        seed=42,
        mixed_precision=False,
        grad_accum_steps=1,
        checkpoint_dir=checkpoint_dir,
        save_interval=1,
        log_dir=None,
        run_name=f"sweep_{pt.key}",
        jsonl_log=jsonl_log,
        eval_every=0,
        nan_tolerance=5,
    )


def _parse_jsonl_metrics(jsonl_path: str) -> dict[str, float]:
    """Parse a JSONL log and return epoch-averaged metrics (step records only)."""
    path = Path(jsonl_path)
    if not path.is_file():
        return {}

    step_records: list[dict] = []
    diag_records: list[dict] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Diagnostic records have cos_sim_score_ctrl; step records have total_loss.
            if "cos_sim_score_ctrl" in rec:
                diag_records.append(rec)
            elif "total_loss" in rec:
                step_records.append(rec)

    result: dict[str, float] = {}

    scalar_keys = ["total_loss", "dsm_loss", "control_energy", "path_kl",
                   "quality_loss", "jarzynski", "grad_norm_score", "grad_norm_control"]
    for key in scalar_keys:
        vals = [r[key] for r in step_records if key in r and r[key] is not None]
        if vals:
            result[key] = float(sum(vals) / len(vals))

    if diag_records:
        last_diag = diag_records[-1]
        for k in ["cos_sim_score_ctrl", "score_drift_mag", "ctrl_drift_mag", "ctrl_score_mag_ratio"]:
            if k in last_diag:
                result[k] = float(last_diag[k])

    return result


def _run_eval(
    checkpoint_path: str,
    sde_config: ScoreSDEConfig,
    device: str,
    num_samples: int = 256,
) -> dict[str, float]:
    """Load a combined checkpoint and evaluate FID/IS."""
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild from saved configs if present.
    score_cfg_dict = state.get("score_config")
    ctrl_cfg_dict = state.get("control_config")

    if score_cfg_dict is not None:
        score_model = build_score_model(ScoreUNetConfig(**score_cfg_dict))
    else:
        score_model = build_score_model(ScoreUNetConfig(in_channels=1, base_channels=32))

    score_sd = state.get("score_state_dict") or state.get("model")
    if score_sd is not None:
        score_model.load_state_dict(score_sd, strict=True)

    # Prefer EMA weights for evaluation.
    ema_sd = state.get("ema_state_dict")
    if ema_sd is not None:
        # EMA state_dict has shadow_ prefix; strip it.
        ema_weights = {
            k.removeprefix("shadow_"): v
            for k, v in ema_sd.items()
            if k.startswith("shadow_")
        }
        if ema_weights:
            try:
                score_model.load_state_dict(ema_weights, strict=True)
            except Exception:
                pass  # fall back to raw weights already loaded

    if ctrl_cfg_dict is not None:
        ctrl_policy = build_control_policy(ControlConfig(**ctrl_cfg_dict))
    else:
        ctrl_policy = build_control_policy(ControlConfig(state_dim=784, hidden_dim=128, depth=3, time_embedding_dim=8))

    ctrl_sd = state.get("control_state_dict") or state.get("control")
    if ctrl_sd is not None:
        ctrl_policy.load_state_dict(ctrl_sd, strict=True)

    score_model.to(device)
    ctrl_policy.to(device)

    eval_cfg = EvaluationConfig(
        dataset_name="fashionmnist",
        num_samples=num_samples,
        batch_size=64,
        device=device,
    )
    return evaluate_sampler(score_model, ctrl_policy, sde_config, eval_cfg)


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_sweep(
    configs_to_run: Optional[list[str]] = None,
    skip_eval: bool = False,
    output_path: str = "data/results/hyperparam_sweep_fmnist.json",
) -> dict[str, dict]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results so we can resume a partial sweep.
    existing: dict[str, dict] = {}
    if output_file.is_file():
        try:
            with output_file.open() as f:
                existing = json.load(f)
            logger.info("Loaded %d existing results from %s", len(existing), output_file)
        except Exception:
            existing = {}

    points = [p for p in SWEEP_CONFIGS if configs_to_run is None or p.key in configs_to_run]

    sweep_dir = _ROOT / "checkpoints" / "sweep_fmnist"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_dir = _ROOT / "logs" / "sweep_fmnist"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = dict(existing)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for pt in points:
        if pt.key in existing:
            logger.info("Config %s already in results — skipping.", pt.key)
            continue

        logger.info(
            "=== Sweep config %s: control_weight=%.4f path_kl_weight=%.4f quality_weight=%.4f ===",
            pt.key, pt.control_weight, pt.path_kl_weight, pt.quality_weight,
        )

        ckpt_dir = str(sweep_dir / f"config_{pt.key}")
        jsonl_path = str(log_dir / f"sweep_{pt.key}.jsonl")

        # Remove any stale JSONL from a previous partial run.
        if Path(jsonl_path).is_file():
            Path(jsonl_path).unlink()

        train_cfg = _build_training_config(pt, ckpt_dir, jsonl_path)

        t0 = time.time()
        try:
            train_metrics = train_controlled_score(train_cfg)
        except Exception as exc:
            logger.error("Config %s training failed: %s", pt.key, exc)
            all_results[pt.key] = {
                "config": {
                    "control_weight": pt.control_weight,
                    "path_kl_weight": pt.path_kl_weight,
                    "quality_weight": pt.quality_weight,
                    "description": pt.description,
                },
                "error": str(exc),
            }
            _save_results(all_results, output_file)
            continue

        train_wall = time.time() - t0
        logger.info("Config %s training done in %.1fs", pt.key, train_wall)

        # Parse JSONL for epoch-averaged metrics.
        jsonl_metrics = _parse_jsonl_metrics(jsonl_path)

        # Merge training return metrics with JSONL averages.
        # JSONL averages are more reliable (mean over all steps).
        merged_train = {**train_metrics, **jsonl_metrics}

        eval_metrics: dict[str, float] = {}
        if not skip_eval:
            ckpt_candidates = sorted(Path(ckpt_dir).glob("controlled_epoch_*.pt"))
            ckpt_last = Path(ckpt_dir) / "controlled_last.pt"
            eval_ckpt = str(ckpt_last) if ckpt_last.is_file() else (
                str(ckpt_candidates[-1]) if ckpt_candidates else None
            )
            if eval_ckpt:
                logger.info("Config %s: running --quick eval (256 samples) from %s", pt.key, eval_ckpt)
                t_eval = time.time()
                try:
                    sde_cfg = ScoreSDEConfig(
                        beta_min=0.1, beta_max=5.0, num_steps=50, clamp=5.0,
                        control_weight=1.0, sigma_min=0.01, sigma_max=1.0,
                    )
                    eval_metrics = _run_eval(eval_ckpt, sde_cfg, device, num_samples=256)
                    logger.info(
                        "Config %s eval: FID=%.2f IS=%.3f (%.1fs)",
                        pt.key, eval_metrics.get("fid", float("nan")),
                        eval_metrics.get("inception_score_mean", float("nan")),
                        time.time() - t_eval,
                    )
                except Exception as exc:
                    logger.error("Config %s eval failed: %s", pt.key, exc)
                    eval_metrics = {"eval_error": str(exc)}
            else:
                logger.warning("Config %s: no checkpoint found for eval.", pt.key)

        all_results[pt.key] = {
            "config": {
                "control_weight": pt.control_weight,
                "path_kl_weight": pt.path_kl_weight,
                "quality_weight": pt.quality_weight,
                "description": pt.description,
            },
            "train": merged_train,
            "eval": eval_metrics,
            "wall_clock_train_sec": round(train_wall, 1),
        }

        # Save incrementally so a crash doesn't lose all progress.
        _save_results(all_results, output_file)
        logger.info("Config %s results saved.", pt.key)

    logger.info("Sweep complete. Results at: %s", output_file)
    return all_results


def _save_results(results: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU-feasible hyperparameter sweep on FashionMNIST.")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["A", "B", "C", "D", "E", "F"],
        default=None,
        help="Subset of configs to run (default: all A-F).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip FID/IS evaluation; only train.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/hyperparam_sweep_fmnist.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_sweep(
        configs_to_run=args.configs,
        skip_eval=args.skip_eval,
        output_path=args.output,
    )
    # Print summary table.
    print("\n=== Sweep Summary ===")
    print(f"{'Key':<4} {'ctrl_w':<8} {'pkl_w':<8} {'qlt_w':<8} {'DSM loss':<12} {'ctrl_nrg':<12} {'path_kl':<12} {'FID':<10} {'IS':<8}")
    print("-" * 90)
    for key in ["A", "B", "C", "D", "E", "F"]:
        if key not in results:
            continue
        r = results[key]
        if "error" in r:
            print(f"{key:<4} ERROR: {r['error'][:60]}")
            continue
        cfg = r.get("config", {})
        train = r.get("train", {})
        ev = r.get("eval", {})
        print(
            f"{key:<4} {cfg.get('control_weight', '?'):<8} "
            f"{cfg.get('path_kl_weight', '?'):<8} "
            f"{cfg.get('quality_weight', '?'):<8} "
            f"{train.get('dsm_loss', float('nan')):<12.4f} "
            f"{train.get('control_energy', float('nan')):<12.4f} "
            f"{train.get('path_kl', float('nan')):<12.4f} "
            f"{ev.get('fid', float('nan')):<10.2f} "
            f"{ev.get('inception_score_mean', float('nan')):<8.3f}"
        )
