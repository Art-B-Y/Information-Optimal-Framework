from __future__ import annotations

"""Validate all ITS Hydra configuration groups.

Instantiates each config group via ``hydra.compose`` and prints the resolved
config tree.  Run this script to surface config-hierarchy mismatches without
launching a full experiment.

Usage::

    python scripts/validate_configs.py
    python scripts/validate_configs.py --group experiment --name controlled_mnist
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal check that the configs directory exists and is parseable before we
# attempt to import Hydra (which may not be installed in all environments).
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"

KNOWN_TOP_LEVEL_CONFIGS = [
    "config",
    "final_phase",
    "controlled_mnist",
    "controlled_cifar",
    "train_long_run",
]

KNOWN_EXPERIMENT_CONFIGS = [
    "gaussian_mixture",
    "double_well",
    "controlled_mnist",
    "controlled_cifar",
    "final_phase",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ITS Hydra configs.")
    parser.add_argument(
        "--group",
        default=None,
        help="Config group to validate (e.g. 'experiment').  Omit to validate all.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Config name within the group (e.g. 'controlled_mnist').  Omit to validate all in group.",
    )
    parser.add_argument(
        "--no-instantiate",
        action="store_true",
        default=False,
        help="Only resolve the config tree; do not call instantiate() on experiment targets.",
    )
    return parser.parse_args()


def _check_configs_dir() -> bool:
    if not CONFIGS_DIR.exists():
        print(f"[ERROR] configs/ directory not found at: {CONFIGS_DIR}", file=sys.stderr)
        return False
    experiment_dir = CONFIGS_DIR / "experiment"
    if not experiment_dir.exists():
        print(f"[ERROR] configs/experiment/ directory not found at: {experiment_dir}", file=sys.stderr)
        return False
    return True


def validate_config(config_name: str, overrides: list[str] | None = None, instantiate_exp: bool = True) -> bool:
    """Resolve *config_name* via Hydra and optionally instantiate the experiment.

    Returns True on success, False on failure.
    """
    try:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:
        print(f"[ERROR] Hydra is not installed: {exc}", file=sys.stderr)
        return False

    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIGS_DIR),
            job_name="its-validate",
        ):
            cfg = compose(config_name=config_name, overrides=overrides or [])

        print(f"\n{'=' * 60}")
        print(f"Config: {config_name}  overrides={overrides or []}")
        print(OmegaConf.to_yaml(cfg))

        if instantiate_exp and hasattr(cfg, "experiment"):
            print(f"  [instantiate] Calling instantiate(cfg.experiment) ...")
            bundle = instantiate(cfg.experiment)
            keys = sorted(bundle.keys()) if isinstance(bundle, dict) else ["(non-dict result)"]
            print(f"  [instantiate] OK — bundle keys: {keys}")

        print(f"[OK] {config_name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {config_name}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    args = parse_args()

    if not _check_configs_dir():
        return 1

    # Build the list of (top_level_config_name, overrides) pairs to validate.
    pairs: list[tuple[str, list[str]]] = []

    if args.group == "experiment" or args.group is None:
        names = [args.name] if args.name else KNOWN_EXPERIMENT_CONFIGS
        for name in names:
            yaml_path = CONFIGS_DIR / "experiment" / f"{name}.yaml"
            if not yaml_path.exists():
                print(f"[WARN] experiment/{name}.yaml not found — skipping", file=sys.stderr)
                continue
            # Each experiment config is validated by composing a top-level config
            # that sets the experiment group.  Use "config" as the base if a
            # dedicated top-level config for this experiment doesn't exist.
            top = name if (CONFIGS_DIR / f"{name}.yaml").exists() else "config"
            override = f"experiment={name}"
            pairs.append((top, [override]))

    if args.group is None:
        # Also validate every dedicated top-level config.
        for name in KNOWN_TOP_LEVEL_CONFIGS:
            yaml_path = CONFIGS_DIR / f"{name}.yaml"
            if not yaml_path.exists():
                print(f"[WARN] {name}.yaml not found — skipping", file=sys.stderr)
                continue
            if not any(cfg_name == name for cfg_name, _ in pairs):
                pairs.append((name, []))

    if not pairs:
        print("No configs to validate.", file=sys.stderr)
        return 1

    failures = 0
    for cfg_name, overrides in pairs:
        ok = validate_config(cfg_name, overrides=overrides, instantiate_exp=not args.no_instantiate)
        if not ok:
            failures += 1

    print(f"\n{'=' * 60}")
    total = len(pairs)
    passed = total - failures
    print(f"Results: {passed}/{total} configs OK, {failures} failures.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
