"""watch_training.py — Live tail of a JSONL training log.

Prints a rolling summary table of the last N recorded steps, refreshed every
REFRESH_SECONDS seconds.  Automatically finds the most recently modified
*.jsonl file under `logs/` if no path is given.

Usage
-----
    python scripts/watch_training.py                         # auto-find latest log
    python scripts/watch_training.py logs/ctrl_run.jsonl     # explicit path
    python scripts/watch_training.py --rows 20 --interval 3  # custom settings
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


METRICS = [
    "epoch",
    "global_step",
    "total_loss",
    "dsm_loss",
    "control_energy",
    "path_kl",
    "quality_loss",
    "jarzynski",
    "grad_norm_score",
    "grad_norm_control",
    "cos_sim_score_ctrl",
    "ctrl_score_mag_ratio",
    "wall_clock_seconds",
]

# Display widths for each column
COL_WIDTHS = {
    "epoch": 5,
    "global_step": 8,
    "total_loss": 10,
    "dsm_loss": 10,
    "control_energy": 14,
    "path_kl": 9,
    "quality_loss": 12,
    "jarzynski": 10,
    "grad_norm_score": 15,
    "grad_norm_control": 17,
    "cos_sim_score_ctrl": 18,
    "ctrl_score_mag_ratio": 19,
    "wall_clock_seconds": 17,
}

SHORT_NAMES = {
    "epoch": "ep",
    "global_step": "step",
    "total_loss": "total",
    "dsm_loss": "dsm",
    "control_energy": "ctrl_energy",
    "path_kl": "path_kl",
    "quality_loss": "quality",
    "jarzynski": "jarzyn",
    "grad_norm_score": "gnorm_score",
    "grad_norm_control": "gnorm_ctrl",
    "cos_sim_score_ctrl": "cos_sim",
    "ctrl_score_mag_ratio": "mag_ratio",
    "wall_clock_seconds": "elapsed_s",
}


def find_latest_jsonl(log_dir: Path) -> Optional[Path]:
    """Return the most recently modified *.jsonl file under *log_dir*."""
    candidates = list(log_dir.glob("**/*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_tail(path: Path, n: int) -> List[dict]:
    """Read the last *n* JSON-line records from *path*.

    Skips malformed lines silently.
    """
    records: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records[-n:]


def fmt_val(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_table(records: List[dict], columns: List[str]) -> str:
    """Render *records* as a fixed-width ASCII table."""
    if not records:
        return "(no records yet)"

    # Determine which columns actually appear in the data.
    present = [c for c in columns if any(c in r for r in records)]
    if not present:
        return "(records present but contain no recognised metric keys)"

    header_parts = [SHORT_NAMES.get(c, c).center(COL_WIDTHS.get(c, 12)) for c in present]
    sep_parts = ["-" * COL_WIDTHS.get(c, 12) for c in present]
    header = " | ".join(header_parts)
    sep = "-+-".join(sep_parts)

    rows = []
    for rec in records:
        row_parts = [fmt_val(rec.get(c)).center(COL_WIDTHS.get(c, 12)) for c in present]
        rows.append(" | ".join(row_parts))

    return "\n".join([header, sep] + rows)


def clear_screen() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live tail of a JSONL training log.")
    parser.add_argument(
        "log_file",
        nargs="?",
        default=None,
        help="Path to a *.jsonl log file.  If omitted, auto-detects the most recent file under logs/.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of most-recent steps to show (default: 10).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Refresh interval in seconds (default: 5).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to search for *.jsonl files when no explicit path is given (default: logs/).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Print once and exit (non-interactive mode).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]

    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = root / log_path
    else:
        log_dir = root / args.log_dir
        log_path = find_latest_jsonl(log_dir)
        if log_path is None:
            print(f"No *.jsonl files found under {log_dir}.  Pass an explicit path.", file=sys.stderr)
            return 1

    print(f"Watching: {log_path}  (refresh every {args.interval}s, showing last {args.rows} steps)")
    print("Press Ctrl-C to stop.\n")

    try:
        while True:
            records = read_tail(log_path, args.rows)
            clear_screen()
            print(f"Watching: {log_path}")
            print(f"Updated : {time.strftime('%H:%M:%S')}  |  records so far: {_count_lines(log_path)}\n")
            print(render_table(records, METRICS))
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
