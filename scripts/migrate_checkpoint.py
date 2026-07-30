"""Migrate pre-Phase-1 score checkpoints to include model_config.

Old checkpoints (score_epoch_0001.pt through score_epoch_0050.pt) were saved
before the architecture-config saving fix.  This script:

1. Loads each checkpoint.
2. Infers the model architecture from the parameter tensor shapes in the
   checkpoint's state dict.
3. Re-saves the checkpoint with a ``model_config`` key attached.
4. Also normalises key names: adds ``model_state_dict`` and
   ``optimizer_state_dict`` aliases alongside the legacy ``model`` / ``optimizer``.

Architecture inference logic:
  - The first conv layer is ``down_blocks.0.net.2`` (a Conv2d).
  - Its weight shape is ``(out_ch, in_ch, kH, kW)``.
  - ``in_ch`` gives ``in_channels``.
  - ``out_ch`` gives ``base_channels * channel_mults[0]`` (always mult=1 at idx 0).
  - ``channel_mults`` can be inferred from the number of down-block groups.

Usage::

    python scripts/migrate_checkpoint.py                         # migrates all score_epoch_*.pt
    python scripts/migrate_checkpoint.py --dir checkpoints/score # explicit directory
    python scripts/migrate_checkpoint.py --dry-run               # preview only, no writes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _infer_model_config(state_dict: dict) -> dict:
    """Infer ScoreUNetConfig from parameter shapes in *state_dict*."""
    # First conv in the encoder: down_blocks.0.net.2.weight -> (out_ch, in_ch, kH, kW)
    first_conv_key = "down_blocks.0.net.2.weight"
    if first_conv_key not in state_dict:
        raise ValueError(
            f"Cannot infer architecture: key '{first_conv_key}' not found in state dict.\n"
            f"Available keys (first 10): {list(state_dict.keys())[:10]}"
        )
    first_conv_w = state_dict[first_conv_key]
    out_ch0, in_channels = int(first_conv_w.shape[0]), int(first_conv_w.shape[1])

    # base_channels = out_ch0 / channel_mults[0]; mults[0] is always 1
    base_channels = out_ch0

    # Count how many down-blocks exist by looking for down_blocks.N.net.2.weight
    num_blocks = 0
    for key in state_dict:
        if key.startswith("down_blocks.") and key.endswith(".net.2.weight"):
            idx = int(key.split(".")[1])
            num_blocks = max(num_blocks, idx + 1)

    # Infer channel_mults by looking at the output channels of each block
    channel_mults = []
    for i in range(num_blocks):
        key = f"down_blocks.{i}.net.2.weight"
        ch = int(state_dict[key].shape[0])
        channel_mults.append(ch // base_channels)

    return {
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channel_mults": channel_mults,
        "dropout": 0.0,
    }


def migrate_checkpoint(path: Path, dry_run: bool = False) -> bool:
    """Add model_config and canonical key names to *path* in-place.

    Returns True if the checkpoint was modified, False if it was already up-to-date.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if "model_config" in checkpoint and "model_state_dict" in checkpoint:
        print(f"  [skip]  {path.name}  — already up-to-date")
        return False

    modified = False

    # Infer and attach model_config if missing.
    if "model_config" not in checkpoint:
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
        if state_dict is None:
            print(f"  [error] {path.name} — no model state dict found, skipping", file=sys.stderr)
            return False
        try:
            model_cfg = _infer_model_config(state_dict)
            checkpoint["model_config"] = model_cfg
            print(f"  [config] Inferred config for {path.name}: {model_cfg}")
            modified = True
        except Exception as exc:
            print(f"  [error] {path.name} — could not infer config: {exc}", file=sys.stderr)
            return False

    # Add canonical key aliases if missing.
    if "model_state_dict" not in checkpoint and "model" in checkpoint:
        checkpoint["model_state_dict"] = checkpoint["model"]
        modified = True

    if "optimizer_state_dict" not in checkpoint and "optimizer" in checkpoint:
        checkpoint["optimizer_state_dict"] = checkpoint["optimizer"]
        modified = True

    if "ema_state_dict" not in checkpoint and "ema" in checkpoint:
        checkpoint["ema_state_dict"] = checkpoint["ema"]
        modified = True

    if not modified:
        print(f"  [skip]  {path.name}  — no changes needed")
        return False

    if dry_run:
        print(f"  [dry-run] Would update {path.name}")
        return True

    torch.save(checkpoint, path)
    print(f"  [saved] {path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate ITS score checkpoints to include model_config.")
    parser.add_argument("--dir", default="checkpoints/score", help="Directory containing score_epoch_*.pt files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    args = parser.parse_args()

    ckpt_dir = Path(args.dir)
    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}", file=sys.stderr)
        return 1

    checkpoints = sorted(ckpt_dir.glob("score_epoch_*.pt"))
    if not checkpoints:
        print(f"No score_epoch_*.pt files found in {ckpt_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(checkpoints)} checkpoint(s) in {ckpt_dir}")
    updated = 0
    for ckpt in checkpoints:
        if migrate_checkpoint(ckpt, dry_run=args.dry_run):
            updated += 1

    print(f"\n{'[dry-run] ' if args.dry_run else ''}{updated}/{len(checkpoints)} checkpoint(s) updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
