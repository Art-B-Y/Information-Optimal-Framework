from __future__ import annotations

"""
Collect metrics from local log files and save them as CSV/JSON.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import os

os.environ["ITS_LIGHT_IMPORT"] = "1"

from its.analysis.metrics import (
    ParsedRun,
    discover_logs,
    parse_controlled_training_log,
    parse_eval_log,
    parse_score_training_log,
    parse_toy_training_log,
    save_records_csv,
    save_records_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ITS metrics from logs.")
    parser.add_argument(
        "--type",
        required=True,
        choices=["toy", "score", "controlled", "eval"],
        help="Experiment type to parse.",
    )
    parser.add_argument("--logs", nargs="*", default=[], help="Log files or directories to search.")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.log", "*.out", "*.txt"],
        help="Glob patterns for logs.",
    )
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save combined metrics CSV.")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save combined metrics JSON.")
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Verify the log directory/directories exist and contain at least one "
            "parseable log file before attempting collection.  Exits with code 1 "
            "and a diagnostic message if the check fails."
        ),
    )
    return parser.parse_args()


def _check_log_roots(roots: list[str], patterns: list[str]) -> tuple[bool, str]:
    """Return (ok, diagnostic) after checking that roots exist and hold parseable files."""
    existing_roots = [r for r in roots if Path(r).exists()]
    if not existing_roots:
        return False, (
            f"None of the specified log directories exist: {roots}\n"
            "Create at least one training run first (e.g. python scripts/train_score_model.py)."
        )

    found_any = False
    for root in existing_roots:
        p = Path(root)
        for pattern in patterns:
            if list(p.rglob(pattern)):
                found_any = True
                break
        if found_any:
            break

    if not found_any:
        return False, (
            f"No files matching {patterns} found under: {existing_roots}\n"
            "Make sure training scripts are writing logs to these directories."
        )
    return True, "OK"


def main() -> int:
    args = parse_args()
    roots = args.logs or ["outputs", "logs", "."]

    if args.check:
        ok, diagnostic = _check_log_roots(roots, args.patterns)
        if not ok:
            print(f"[check] FAIL: {diagnostic}", file=sys.stderr)
            return 1
        print("[check] OK — log directory exists and contains matching files.")
        return 0

    log_paths = discover_logs(roots, patterns=args.patterns)
    if not log_paths:
        print(
            "No logs found in the specified locations.\n"
            f"  Searched: {roots}\n"
            f"  Patterns: {args.patterns}\n"
            "Run a training script first to generate logs, or use --logs to point to a different directory.",
            file=sys.stderr,
        )
        # Exit code 1 so callers can detect the empty case (Bug 9 fix).
        return 1

    parsers = {
        "toy": parse_toy_training_log,
        "score": parse_score_training_log,
        "controlled": parse_controlled_training_log,
        "eval": parse_eval_log,
    }
    parse_fn = parsers[args.type]

    all_records: List[dict] = []
    parsed_runs: List[ParsedRun] = []
    for path in log_paths:
        try:
            parsed = parse_fn(path)
            if parsed.records:
                parsed_runs.append(parsed)
                for rec in parsed.records:
                    rec = dict(rec)
                    rec["run_id"] = parsed.run_id
                    rec["log_path"] = str(parsed.path)
                    all_records.append(rec)
        except Exception:
            continue

    if not all_records:
        print(
            "No matching records found.\n"
            f"  Parsed {len(log_paths)} log file(s) as type '{args.type}' but found no parseable entries.\n"
            "Check that --type matches the log format (toy / score / controlled / eval).",
            file=sys.stderr,
        )
        # Exit code 1 for callers that test for successful metric collection (Bug 9 fix).
        return 1

    if args.output_csv:
        save_records_csv(all_records, Path(args.output_csv))
    if args.output_json:
        save_records_json(all_records, Path(args.output_json))

    print(f"Parsed {len(parsed_runs)} runs, {len(all_records)} records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
