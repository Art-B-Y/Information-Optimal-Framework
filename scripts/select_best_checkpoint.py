"""Select the best checkpoint from a JSONL training log.

Selection criterion (Session 5, Step 2C):
  Primary:   lowest path_kl among eligible epochs
  Secondary: only consider epochs where dsm_loss is within 5% of its minimum
             across the log (ensures score model quality is not severely degraded)

Usage
-----
    python scripts/select_best_checkpoint.py \\
        --jsonl logs/suite_config_D_5ep.jsonl \\
        --checkpoint-dir checkpoints/suite/config_D_5ep

Returns the path of the best checkpoint and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _load_epoch_summaries(jsonl_path: Path) -> list[dict]:
    """Read a JSONL training log and aggregate per-epoch mean metrics.

    JSONL records that have an 'epoch' and 'path_kl' field are included.
    Returns a list of dicts, one per epoch, sorted by epoch number.
    """
    per_epoch: dict[int, dict[str, list[float]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = record.get("epoch")
            if epoch is None:
                continue
            epoch = int(epoch)
            if epoch not in per_epoch:
                per_epoch[epoch] = {"dsm_loss": [], "path_kl": [], "control_energy": [], "quality_loss": []}
            for key in per_epoch[epoch]:
                if key in record and record[key] is not None:
                    try:
                        per_epoch[epoch][key].append(float(record[key]))
                    except (TypeError, ValueError):
                        pass

    summaries = []
    for epoch in sorted(per_epoch):
        row: dict = {"epoch": epoch}
        for key, vals in per_epoch[epoch].items():
            if vals:
                row[key] = sum(vals) / len(vals)
        summaries.append(row)
    return summaries


def select_best(
    jsonl_path: Path,
    checkpoint_dir: Path,
    dsm_tolerance: float = 0.05,
) -> Optional[Path]:
    """Return the path of the best checkpoint.

    Selects the epoch with the lowest path_kl among epochs where dsm_loss
    is within *dsm_tolerance* (5% by default) of the minimum dsm_loss in the log.
    """
    summaries = _load_epoch_summaries(jsonl_path)
    if not summaries:
        print(f"No per-epoch records found in {jsonl_path}", file=sys.stderr)
        return None

    # Filter to epochs with both path_kl and dsm_loss recorded.
    valid = [s for s in summaries if "path_kl" in s and "dsm_loss" in s]
    if not valid:
        print("No records with both path_kl and dsm_loss found.", file=sys.stderr)
        return None

    min_dsm = min(s["dsm_loss"] for s in valid)
    dsm_threshold = min_dsm * (1.0 + dsm_tolerance)
    eligible = [s for s in valid if s["dsm_loss"] <= dsm_threshold]

    if not eligible:
        # Fallback: use all valid epochs.
        eligible = valid

    best = min(eligible, key=lambda s: s["path_kl"])
    best_epoch = best["epoch"]

    print(f"{'Epoch':>6}  {'DSM loss':>10}  {'path_kl':>10}  {'ctrl_energy':>12}  {'eligible':>8}")
    print("-" * 58)
    for s in valid:
        flag = "← BEST" if s["epoch"] == best_epoch else ("*" if s["dsm_loss"] <= dsm_threshold else "")
        print(f"{s['epoch']:>6}  {s.get('dsm_loss', float('nan')):>10.4f}  "
              f"{s.get('path_kl', float('nan')):>10.4f}  "
              f"{s.get('control_energy', float('nan')):>12.6f}  {flag}")

    print(f"\nBest epoch: {best_epoch}  (path_kl={best['path_kl']:.4f}, dsm_loss={best['dsm_loss']:.4f})")
    print(f"DSM threshold: min_dsm={min_dsm:.4f}  ×{1+dsm_tolerance:.2f} = {dsm_threshold:.4f}")

    # Resolve checkpoint path.
    ckpt_candidates = [
        checkpoint_dir / f"controlled_epoch_{best_epoch:04d}.pt",
        checkpoint_dir / f"controlled_epoch_{best_epoch}.pt",
    ]
    for c in ckpt_candidates:
        if c.is_file():
            print(f"Best checkpoint: {c}")
            return c

    # Fall back to latest checkpoint.
    latest = checkpoint_dir / "controlled_last.pt"
    if latest.is_file():
        print(f"Epoch checkpoint not found; using latest: {latest}")
        return latest

    print(f"No checkpoint found in {checkpoint_dir}", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the best checkpoint from a JSONL training log.")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL training log.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing epoch checkpoints.")
    parser.add_argument(
        "--dsm-tolerance",
        type=float,
        default=0.05,
        help="Fraction above minimum DSM loss still considered eligible (default 0.05 = 5%%).",
    )
    parser.add_argument("--output", default=None, help="Write best checkpoint path to this file.")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"ERROR: JSONL log not found: {jsonl_path}", file=sys.stderr)
        return 1

    ckpt_dir = Path(args.checkpoint_dir)
    best = select_best(jsonl_path, ckpt_dir, dsm_tolerance=args.dsm_tolerance)

    if best is None:
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(str(best))
        print(f"Path written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
