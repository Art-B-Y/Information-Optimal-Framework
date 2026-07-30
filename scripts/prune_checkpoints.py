"""Checkpoint retention policy pruner (Step 3D, Session Speed).

Retention policy:
  - Keep ALL checkpoints from the last 5 epochs.
  - Keep every 10th epoch checkpoint before that.
  - Delete everything else.

Recognised filename pattern: ``controlled_epoch_{NNNN}.pt`` or
``score_epoch_{NNNN}.pt``.  The ``controlled_last.pt`` and ``autosave.pt``
files are always preserved.

Usage:
    python scripts/prune_checkpoints.py --ckpt-dir checkpoints/controlled_v2_5a_seed42
    python scripts/prune_checkpoints.py --ckpt-dir checkpoints/score_fmnist_v2 --dry-run
    python scripts/prune_checkpoints.py --ckpt-dir ... --keep-last 5 --keep-every 10
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_EPOCH_PATTERN = re.compile(r"(?:controlled|score)_epoch_(\d+)\.pt$")
_ALWAYS_KEEP = {"controlled_last.pt", "score_best.pt", "autosave.pt",
                "autosave.pt.tmp", "ipf_last.pt"}


def _find_epoch_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch_int, path)] sorted by epoch ascending."""
    pairs = []
    for p in ckpt_dir.iterdir():
        m = _EPOCH_PATTERN.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    return sorted(pairs, key=lambda x: x[0])


def prune(
    ckpt_dir: Path,
    keep_last: int = 5,
    keep_every: int = 10,
    dry_run: bool = False,
) -> dict:
    """Apply retention policy and delete stale checkpoints.

    Parameters
    ----------
    ckpt_dir   : directory containing checkpoint files.
    keep_last  : always keep this many of the most recent epoch checkpoints.
    keep_every : keep every Nth epoch checkpoint before the `keep_last` window.
    dry_run    : report what would be deleted without actually deleting.

    Returns
    -------
    dict with keys ``kept``, ``deleted``, ``total`` (lists of Path strings).
    """
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    epochs = _find_epoch_checkpoints(ckpt_dir)
    if not epochs:
        return {"kept": [], "deleted": [], "total": []}

    all_epochs = [e for e, _ in epochs]
    n = len(all_epochs)

    keep_set: set[int] = set()

    # Keep last `keep_last` epochs
    for e, _ in epochs[-keep_last:]:
        keep_set.add(e)

    # Keep every `keep_every`th epoch before the last `keep_last`
    for e, _ in epochs[:-keep_last]:
        if e % keep_every == 0:
            keep_set.add(e)

    kept, deleted = [], []
    for epoch_num, path in epochs:
        if epoch_num in keep_set:
            kept.append(str(path))
        else:
            deleted.append(str(path))
            if not dry_run:
                path.unlink(missing_ok=True)

    return {"kept": sorted(kept), "deleted": sorted(deleted),
            "total": [str(p) for _, p in epochs]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="Directory containing epoch checkpoint files")
    parser.add_argument("--keep-last", type=int, default=5,
                        help="Number of most-recent epoch checkpoints to keep (default 5)")
    parser.add_argument("--keep-every", type=int, default=10,
                        help="Keep every Nth epoch before the keep-last window (default 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be deleted without deleting")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    result = prune(ckpt_dir, keep_last=args.keep_last,
                   keep_every=args.keep_every, dry_run=args.dry_run)

    label = "[DRY RUN] " if args.dry_run else ""
    print(f"{label}Checkpoint pruning: {ckpt_dir}")
    print(f"  Total epoch checkpoints found : {len(result['total'])}")
    print(f"  Kept  : {len(result['kept'])}")
    print(f"  {'Would delete' if args.dry_run else 'Deleted'}: {len(result['deleted'])}")
    if result["deleted"]:
        for p in result["deleted"]:
            print(f"    {'[would delete]' if args.dry_run else '[deleted]'} {Path(p).name}")


if __name__ == "__main__":
    main()
