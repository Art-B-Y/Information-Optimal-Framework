"""Merge JSONL training logs from multiple segmented runs (Session Speed).

When a training run is split into segments by launch_segmented_training.py,
each segment writes its own JSONL log. This script stitches them together
into a single chronological file suitable for plotting and analysis.

Records are passed through in file order. Duplicate steps (where a segment
overlapped with a previous one due to checkpoint reload) are deduplicated by
keeping the first occurrence.

Usage:
    python scripts/stitch_logs.py seg1.jsonl seg2.jsonl -o stitched.jsonl
    python scripts/stitch_logs.py logs/run_5a_seg*.jsonl -o logs/run_5a.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def stitch_jsonl_logs(
    segment_paths: list[Path],
    output_path: Path,
    dedup_key: str = "step",
) -> int:
    """Merge JSONL files from multiple segments into one output file.

    Parameters
    ----------
    segment_paths : ordered list of JSONL files (one per training segment).
    output_path   : destination file; parent directories are created if needed.
    dedup_key     : record field used for deduplication (default "step").
                    Records whose dedup_key was already seen are skipped.

    Returns
    -------
    Number of records written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    written = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        for seg in segment_paths:
            seg = Path(seg)
            if not seg.exists():
                continue
            with seg.open("r", encoding="utf-8") as in_fh:
                for line in in_fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get(dedup_key)
                    if key is not None:
                        if key in seen:
                            continue
                        seen.add(key)
                    out_fh.write(json.dumps(record) + "\n")
                    written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch segmented JSONL training logs")
    parser.add_argument("segments", nargs="+", help="JSONL segment files (in order)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--dedup-key", default="step",
                        help="Record field for deduplication (default: step)")
    args = parser.parse_args()

    paths = [Path(p) for p in args.segments]
    out = Path(args.output)
    n = stitch_jsonl_logs(paths, out, dedup_key=args.dedup_key)
    print(f"Stitched {len(paths)} segments -> {out} ({n} records)")


if __name__ == "__main__":
    main()
