"""Snapshot the full compute environment for reproducibility (Step 5B, Session 7).

Writes environment_snapshot.txt alongside training artifacts.

Usage:
    python scripts/snapshot_environment.py
    python scripts/snapshot_environment.py --output data/results/environment_snapshot.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run(cmd: list[str], default: str = "(unavailable)") -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL,
                                       timeout=30).strip()
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "environment_snapshot.txt"))
    args = parser.parse_args()

    lines: list[str] = ["=== ITS PROJECT ENVIRONMENT SNAPSHOT ===", ""]

    # Git state
    lines.append("--- Git ---")
    lines.append(f"commit: {_run(['git', 'rev-parse', 'HEAD'])}")
    lines.append(f"branch: {_run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}")
    diff = _run(["git", "diff", "--stat"])
    lines.append(f"uncommitted changes: {diff or 'none'}")
    lines.append("")

    # Python
    lines.append("--- Python ---")
    lines.append(f"version: {sys.version}")
    lines.append(f"executable: {sys.executable}")
    lines.append("")

    # pip freeze
    lines.append("--- pip freeze ---")
    lines.append(_run([sys.executable, "-m", "pip", "freeze"]))
    lines.append("")

    # conda list (if available)
    conda = _run(["conda", "list", "--export"])
    if conda != "(unavailable)":
        lines.append("--- conda list ---")
        lines.append(conda)
        lines.append("")

    # CUDA / GPU
    lines.append("--- GPU / CUDA ---")
    try:
        import torch
        lines.append(f"torch version: {torch.__version__}")
        lines.append(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda version: {torch.version.cuda}")
            lines.append(f"device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                lines.append(f"  device {i}: {props.name} ({props.total_memory // 1024**2} MB)")
    except ImportError:
        lines.append("torch not available")
    lines.append(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    lines.append("")

    # CUDA/Torch environment variables
    lines.append("--- Relevant environment variables ---")
    for key, val in sorted(os.environ.items()):
        if any(prefix in key.upper() for prefix in ["CUDA", "TORCH", "NCCL", "PYTHONPATH"]):
            lines.append(f"  {key}={val}")
    lines.append("")

    # nvidia-smi
    lines.append("--- nvidia-smi ---")
    lines.append(_run(["nvidia-smi"]))
    lines.append("")

    out = "\n".join(lines)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out)
    print(f"Environment snapshot written to {out_path}")


if __name__ == "__main__":
    main()
