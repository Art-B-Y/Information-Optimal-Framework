from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import argparse
import sys
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from its.eval import EvaluationConfig, evaluate_sampler
from its.eval.evaluator import evaluate_ddpm_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate controlled score-based sampler (FID/IS/NFE).")
    parser.add_argument("--config-name", "-c", default="controlled_mnist", help="Hydra config to use.")
    parser.add_argument("--override", "-o", action="append", default=[], help="Hydra override (repeatable).")
    parser.add_argument("--overrides", default=None, help="Space-separated Hydra overrides.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help=(
            "Number of generated samples.  Defaults to -1 which uses the dataset-specific "
            "default (5000 for FashionMNIST, 10000 for CIFAR-10).  Use --quick for 256."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for sampling and metrics.")
    parser.add_argument("--device", type=str, default=None, help="Device override for evaluation.")
    parser.add_argument(
        "--model-ckpt",
        type=str,
        default=None,
        help="Path to a score-model checkpoint.  Required when --mode=controlled (unless --baseline-only).",
    )
    parser.add_argument(
        "--control-ckpt",
        type=str,
        default=None,
        help="Path to a control-policy checkpoint.  Required when --mode=controlled (unless --baseline-only).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["controlled", "sde", "ddpm", "ddim"],
        default="controlled",
        help=(
            "Sampler mode. 'controlled'/'sde' require --model-ckpt and --control-ckpt "
            "unless --baseline-only is set. 'ddpm'/'ddim' only require --model-ckpt."
        ),
    )
    # Kept for backward-compat with existing callers that used --baseline.
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["sde", "ddpm", "ddim"],
        default=None,
        help="Deprecated alias for --mode.  Use --mode instead.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        default=False,
        help=(
            "Run evaluation with uninitialised/random weights.  "
            "Skips the checkpoint requirement for --mode=controlled.  "
            "Reported metrics will be meaningless but useful for smoke-testing."
        ),
    )
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file to append results.")
    parser.add_argument("--quick", action="store_true", default=False, help="Override num-samples to 256 for fast iteration.")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for reproducible FID/IS evaluation (Step 6C). "
            "When set, both real-image shuffling and generated samples use this seed."
        ),
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help=(
            "Override num_samples to the dataset-specific full-eval count "
            "(5000 for FashionMNIST/MNIST, 10000 for CIFAR-10) and compute "
            "dual-seed FID variance for publication-quality results (Steps 6D/6E)."
        ),
    )
    return parser.parse_args()


def _is_combined_checkpoint(state: dict) -> bool:
    """Return True if *state* is a combined score+control checkpoint."""
    return isinstance(state, dict) and (
        "score_state_dict" in state or "control_state_dict" in state
    )


def _rebuild_from_config(state: dict) -> tuple:
    """Rebuild (score_model, control_policy) from config keys saved in *state*.

    Returns (score_model, control_policy) or (None, None) if configs are absent.
    """
    import dataclasses
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy

    score_model = None
    control_policy = None

    score_cfg_dict = state.get("score_config") or state.get("model_config")
    if score_cfg_dict is not None:
        try:
            score_cfg = ScoreUNetConfig(**score_cfg_dict)
            score_model = build_score_model(score_cfg)
        except Exception:
            score_model = None

    ctrl_cfg_dict = state.get("control_config")
    if ctrl_cfg_dict is not None:
        try:
            ctrl_cfg = ControlConfig(**ctrl_cfg_dict)
            control_policy = build_control_policy(ctrl_cfg)
        except Exception:
            control_policy = None

    return score_model, control_policy


def maybe_load_checkpoint(
    path: Optional[str],
    score_model: torch.nn.Module,
    control_model: Optional[torch.nn.Module] = None,
) -> tuple:
    """Load a combined or split checkpoint into *score_model* (and optionally *control_model*).

    Returns (score_model, control_model) — may be rebuilt from saved configs if architecture
    differs from the Hydra-instantiated models.
    """
    if path is None:
        return score_model, control_model
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        # If saved configs are present, rebuild models to avoid architecture mismatch.
        rebuilt_score, rebuilt_control = _rebuild_from_config(state)
        if rebuilt_score is not None:
            score_model = rebuilt_score
        if rebuilt_control is not None and control_model is not None:
            control_model = rebuilt_control

        # Load state dicts using canonical keys first, fall back to legacy aliases.
        score_sd = state.get("score_state_dict") or state.get("model_state_dict") or state.get("model")
        if score_sd is not None:
            score_model.load_state_dict(score_sd, strict=True)
        else:
            score_model.load_state_dict(state, strict=True)

        if control_model is not None:
            ctrl_sd = state.get("control_state_dict") or state.get("control")
            if ctrl_sd is not None:
                control_model.load_state_dict(ctrl_sd, strict=True)
    else:
        score_model.load_state_dict(state)

    # Diagnostic: print mean absolute weight of first conv layer after loading.
    # A randomly initialised model and a trained model differ substantially here.
    # If this value is near the random-init value, the checkpoint may not contain
    # trained weights (Audit Category 1, Session 5).
    _diag_conv_keys = [k for k, v in score_model.state_dict().items()
                       if 'weight' in k and len(v.shape) >= 2 and v.shape[-1] > 1]
    if _diag_conv_keys:
        _w = score_model.state_dict()[_diag_conv_keys[0]]
        print(f"[checkpoint-diag] score model '{_diag_conv_keys[0]}' "
              f"mean|w|={_w.abs().mean():.5f}  (random-init ≈ Kaiming ~ 0.17 for 3×3 conv)")
    if control_model is not None:
        _ctrl_keys = [k for k, v in control_model.state_dict().items()
                      if 'weight' in k and len(v.shape) >= 2]
        if _ctrl_keys:
            _cw = control_model.state_dict()[_ctrl_keys[0]]
            print(f"[checkpoint-diag] control '{_ctrl_keys[0]}' "
                  f"mean|w|={_cw.abs().mean():.5f}")

    return score_model, control_model


def maybe_load_state(path: Optional[str], module: torch.nn.Module, key: Optional[str] = None) -> None:
    """Load a bare state-dict into *module*.

    If the file is a combined checkpoint, *key* selects which sub-dict to use
    ('model'/'score_state_dict' for score model, 'control'/'control_state_dict' for control).
    Falls back to the full dict if neither canonical nor legacy key is found.
    """
    if path is None:
        return
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if key:
            # Caller specified which key to use.
            state = state.get(key, state)
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model" in state:
            state = state["model"]
        # else: assume the dict IS the state dict
    module.load_state_dict(state)


def main() -> int:
    args = parse_args()

    # --baseline is a deprecated alias for --mode.
    mode = args.mode
    if args.baseline is not None:
        mode = args.baseline

    # Quick flag overrides sample count for fast iteration (Improvement 4).
    # -1 means "use the dataset-specific default" (5000 for FashionMNIST, 10000 for CIFAR-10).
    # --full-eval overrides both --quick and --num-samples (Step 6D).
    num_samples = 256 if args.quick else args.num_samples

    # Enforce checkpoint requirements for controlled/SDE mode (Bug 4 fix).
    controlled_mode = mode in {"controlled", "sde"}
    if controlled_mode and not args.baseline_only:
        missing = []
        if args.model_ckpt is None:
            missing.append("--model-ckpt")
        if args.control_ckpt is None:
            missing.append("--control-ckpt")
        if missing:
            raise ValueError(
                f"Mode '{mode}' requires explicit checkpoint paths but {missing} were not provided.\n"
                "Pass both --model-ckpt and --control-ckpt, or use --baseline-only to run with "
                "random/uninitialised weights (metrics will be meaningless)."
            )

    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    hydra_overrides: List[str] = []
    if args.override:
        hydra_overrides.extend(args.override)
    if args.overrides:
        hydra_overrides.extend(args.overrides.split())

    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="its-eval"):
        cfg: DictConfig = compose(config_name=args.config_name, overrides=hydra_overrides)

    bundle = instantiate(cfg.experiment)
    score_model = bundle["model"]
    control_policy = bundle["control"]
    training_cfg = bundle["training"]
    sde_cfg = bundle["sde"]

    # Load checkpoints.
    # When both paths point to the same combined checkpoint (typical for controlled mode),
    # use maybe_load_checkpoint which correctly handles all key variants and rebuilds
    # models from saved architecture configs to avoid shape mismatches.
    model_path = Path(args.model_ckpt) if args.model_ckpt else None
    control_path = Path(args.control_ckpt) if args.control_ckpt else None
    if model_path is not None and model_path == control_path:
        # Combined checkpoint: one file holds both models.
        score_model, control_policy = maybe_load_checkpoint(
            args.model_ckpt, score_model, control_policy
        )
    elif model_path is not None and control_path is not None:
        # Separate files: load score model from model_ckpt, control from control_ckpt.
        maybe_load_state(args.model_ckpt, score_model, key="model_state_dict")
        maybe_load_state(args.control_ckpt, control_policy, key="control_state_dict")
    elif model_path is not None:
        # Score-only checkpoint (DDPM/DDIM mode).
        # Use maybe_load_checkpoint to rebuild from saved model_config if present.
        score_model, _ = maybe_load_checkpoint(args.model_ckpt, score_model, None)

    device_str = args.device or getattr(training_cfg, "device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    score_model.to(device_str)
    control_policy.to(device_str)

    eval_cfg = EvaluationConfig(
        dataset_name=getattr(training_cfg, "dataset_name", "fashionmnist"),
        num_samples=num_samples,
        batch_size=args.batch_size,
        device=device_str,
        seed=args.eval_seed,          # Step 6C
        full_eval=args.full_eval,     # Step 6D/6E
    )

    if mode in {"ddpm", "ddim"}:
        results = evaluate_ddpm_baseline(score_model, sde_cfg, eval_cfg, baseline=mode)
    else:
        results = evaluate_sampler(score_model, control_policy, sde_cfg, eval_cfg)

    print("Evaluation results:", results)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"Evaluation results: {results}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
