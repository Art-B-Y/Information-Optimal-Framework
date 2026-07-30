"""Segmented training launcher (Step 4A, Session Speed).

Runs a training script in segments of up to `max_segment_hours` wall-clock
time.  Each segment is a subprocess; when it exits with code 42, the launcher
restarts it from the latest checkpoint.  Any other exit code is treated as an
error and the launcher stops.

The training script must support:
  --max-wall-hours FLOAT  : exit cleanly with code 42 at the time limit.
  --resume-latest         : resume from checkpoints/{run_name}/autosave.pt
                            or controlled_last.pt.

Usage:
    python scripts/launch_segmented_training.py \\
        --script scripts/train_redesigned_v2.py \\
        --run-id 5a \\
        --target-epochs 100 \\
        --checkpoint-dir checkpoints/controlled_v2_5a_seed42 \\
        --max-segment-hours 4

    # Short test (segments of 36 seconds):
    python scripts/launch_segmented_training.py \\
        --script scripts/train_redesigned_v2.py \\
        --run-id 5a --target-epochs 3 --max-segment-hours 0.01
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SEG_EXIT_CODE = 42


def _count_completed_epochs(ckpt_dir: Path) -> int:
    """Estimate completed epochs from the latest checkpoint file."""
    import re
    if not ckpt_dir.is_dir():
        return 0
    # Check autosave first (has epoch field); ipf_last.pt stores epoch=ipf_iteration.
    for fname in ("autosave.pt", "controlled_last.pt", "ipf_last.pt", "score_last.pt"):
        p = ckpt_dir / fname
        if p.exists():
            try:
                import torch
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                ep = int(ckpt.get("epoch", 0))
                if ep > 0:
                    return ep
            except Exception:
                pass
    # Fall back to scanning epoch checkpoint filenames (controlled or score training).
    import re as _re
    max_epoch = 0
    for p in ckpt_dir.iterdir():
        m = _re.match(r"(?:controlled|score)_epoch_(\d+)\.pt", p.name)
        if m:
            max_epoch = max(max_epoch, int(m.group(1)))
    return max_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True,
                        help="Training script path (e.g. scripts/train_redesigned_v2.py)")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier forwarded to the training script (--run 5a)")
    parser.add_argument("--target-epochs", type=int, required=True,
                        help="Total epoch count that defines completion")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Checkpoint directory to watch for progress")
    parser.add_argument("--max-segment-hours", type=float, default=4.0,
                        help="Max wall-clock hours per segment (default 4)")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional arguments forwarded verbatim to the training script")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    script = Path(args.script)
    if not script.exists():
        print(f"ERROR: Training script not found: {script}")
        sys.exit(1)

    segment = 0
    launcher_start = time.time()
    epoch_times: list[tuple[int, float]] = []  # (completed_epochs, wall_seconds)

    print(f"=== Segmented Training Launcher ===")
    print(f"Script       : {script}")
    print(f"Target epochs: {args.target_epochs}")
    print(f"Segment limit: {args.max_segment_hours:.2f} h")
    print(f"Checkpoint   : {ckpt_dir}")

    while True:
        completed_before = _count_completed_epochs(ckpt_dir)
        remaining = args.target_epochs - completed_before
        if remaining <= 0:
            print(f"\nTarget of {args.target_epochs} epochs reached. Done.")
            break

        segment += 1
        elapsed_total = time.time() - launcher_start
        pct = 100.0 * completed_before / args.target_epochs

        if segment > 1 and epoch_times:
            rate = epoch_times[-1][0] / epoch_times[-1][1]  # epochs/sec
            eta_h = remaining / rate / 3600 if rate > 0 else float("inf")
            n_segs = int(eta_h / args.max_segment_hours) + 1
            print(f"\nSegment {segment} starting. "
                  f"Progress: {completed_before}/{args.target_epochs} epochs ({pct:.1f}%). "
                  f"ETA: {eta_h:.1f} h (~{n_segs} segments of {args.max_segment_hours:.1f} h).")
        else:
            print(f"\nSegment {segment} starting. "
                  f"Progress: {completed_before}/{args.target_epochs} ({pct:.1f}%).")

        cmd = [sys.executable, str(script)]
        if args.run_id:
            cmd += ["--run", args.run_id]
        cmd += ["--resume-latest"]
        cmd += [f"--max-wall-hours={args.max_segment_hours}"]
        if args.extra_args:
            cmd += args.extra_args

        seg_start = time.time()
        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(ROOT))
        seg_elapsed = time.time() - seg_start

        completed_after = _count_completed_epochs(ckpt_dir)
        epochs_this_seg = completed_after - completed_before
        epoch_times.append((max(epochs_this_seg, 1), seg_elapsed))

        if result.returncode == _SEG_EXIT_CODE:
            print(f"  Segment {segment} ended cleanly (exit 42) after {seg_elapsed/3600:.2f} h. "
                  f"Completed {epochs_this_seg} epochs this segment.")
            # Prune checkpoints to save disk space
            try:
                prune_cmd = [sys.executable, str(ROOT / "scripts" / "prune_checkpoints.py"),
                             "--ckpt-dir", str(ckpt_dir)]
                subprocess.run(prune_cmd, check=False, capture_output=True)
            except Exception:
                pass
            continue
        elif result.returncode == 0:
            print(f"  Segment {segment} completed normally. Training finished.")
            break
        else:
            print(f"  ERROR: segment {segment} exited with code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    total_elapsed = time.time() - launcher_start
    completed_final = _count_completed_epochs(ckpt_dir)
    print(f"\n=== Launcher finished ===")
    print(f"  Total segments   : {segment}")
    print(f"  Total wall time  : {total_elapsed/3600:.2f} h")
    print(f"  Completed epochs : {completed_final}/{args.target_epochs}")

    summary = {
        "script": str(script),
        "run_id": args.run_id,
        "target_epochs": args.target_epochs,
        "completed_epochs": completed_final,
        "total_segments": segment,
        "total_wall_hours": total_elapsed / 3600,
        "max_segment_hours": args.max_segment_hours,
        "checkpoint_dir": str(ckpt_dir),
    }
    out = ROOT / "data" / "results" / "segmented_run_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"  Summary -> {out}")


if __name__ == "__main__":
    main()
