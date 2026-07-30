"""Run a YAML-defined suite of training experiments sequentially.

Each run entry specifies a Hydra config name, override strings, epoch count,
and whether to run evaluation afterward.  Failures are caught and logged so
the suite continues to the next run rather than aborting.

Usage
-----
    python scripts/run_experiment_suite.py --suite conf/suites/fmnist_config_sweep.yaml
    python scripts/run_experiment_suite.py --suite conf/suites/fmnist_config_sweep.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _load_suite(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    assert "runs" in suite, "Suite YAML must have a top-level 'runs' list."
    return suite


def _run_training(
    run_cfg: dict[str, Any],
    dry_run: bool,
    device: str,
) -> dict[str, Any]:
    """Execute one training run; returns result dict."""
    name = run_cfg.get("name", "unnamed")
    config_name = run_cfg.get("config_name", "controlled_mnist")
    overrides = run_cfg.get("overrides", [])
    epochs = run_cfg.get("epochs", 5)
    eval_after = run_cfg.get("eval_after", False)
    checkpoint_dir = run_cfg.get("checkpoint_dir", f"checkpoints/suite/{name}")
    jsonl_log = run_cfg.get("jsonl_log", f"logs/suite_{name}.jsonl")

    # Build training override list
    all_overrides = list(overrides) + [
        f"experiment.training.epochs={epochs}",
        f"experiment.training.checkpoint_dir={checkpoint_dir}",
        f"experiment.training.jsonl_log={jsonl_log}",
    ]

    cmd = [
        sys.executable, "scripts/train_controlled_score.py",
        "--config-name", config_name,
        "--device", device,
    ] + ["--override=" + o for o in all_overrides]

    result: dict[str, Any] = {
        "name": name,
        "config_name": config_name,
        "overrides": all_overrides,
        "epochs": epochs,
        "checkpoint_dir": checkpoint_dir,
        "jsonl_log": jsonl_log,
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "duration_sec": None,
        "exit_code": None,
        "error": None,
        "eval": None,
    }

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        result["status"] = "dry_run"
        return result

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result["start_time"] = _timestamp()
    try:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=False,
            text=True,
            timeout=run_cfg.get("timeout_sec", 86400),  # 24 h default
        )
        result["exit_code"] = proc.returncode
        result["status"] = "ok" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Training exceeded timeout_sec limit."
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["end_time"] = _timestamp()
    result["duration_sec"] = round(time.time() - t0, 1)

    # Optional evaluation step
    if eval_after and result["status"] == "ok":
        ckpt_last = Path(checkpoint_dir) / "controlled_last.pt"
        eval_seed = run_cfg.get("eval_seed", 42)
        num_samples = run_cfg.get("eval_num_samples", 2048)
        eval_cmd = [
            sys.executable, "scripts/eval_samples.py",
            "--config-name", config_name,
            "--mode", "controlled",
            "--model-ckpt", str(ckpt_last),
            "--control-ckpt", str(ckpt_last),
            "--eval-seed", str(eval_seed),
            "--num-samples", str(num_samples),
            "--device", device,
        ]
        print(f"  Evaluating: {' '.join(eval_cmd)}")
        try:
            eval_proc = subprocess.run(
                eval_cmd,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if eval_proc.returncode == 0:
                # Parse FID from output
                for line in eval_proc.stdout.splitlines():
                    if "fid" in line.lower() or "Evaluation results" in line:
                        result["eval"] = {"raw_output": line.strip()}
                        break
            else:
                result["eval"] = {"error": eval_proc.stderr[-500:]}
        except Exception as exc:
            result["eval"] = {"error": str(exc)}

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YAML suite of ITS training experiments.")
    parser.add_argument("--suite", required=True, help="Path to suite YAML file.")
    parser.add_argument("--device", default="cpu", help="Device for all runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for suite log JSON. Defaults to data/results/suite_log_{timestamp}.json.",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite)
    if not suite_path.is_file():
        print(f"ERROR: Suite file not found: {suite_path}", file=sys.stderr)
        return 1

    suite = _load_suite(suite_path)
    suite_name = suite.get("name", suite_path.stem)
    runs = suite["runs"]

    output_path = Path(args.output) if args.output else Path(f"data/results/suite_log_{_timestamp()}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== ITS Experiment Suite: {suite_name} ===")
    print(f"Runs: {len(runs)}")
    print(f"Device: {args.device}")
    print(f"Log: {output_path}")
    print()

    suite_log: dict[str, Any] = {
        "suite_name": suite_name,
        "suite_file": str(suite_path),
        "start_time": _timestamp(),
        "device": args.device,
        "dry_run": args.dry_run,
        "runs": [],
    }

    for i, run_cfg in enumerate(runs):
        name = run_cfg.get("name", f"run_{i}")
        print(f"[{i+1}/{len(runs)}] {name}")
        result = _run_training(run_cfg, dry_run=args.dry_run, device=args.device)
        suite_log["runs"].append(result)
        status_icon = "✓" if result["status"] == "ok" else "✗"
        dur = f" ({result['duration_sec']}s)" if result["duration_sec"] is not None else ""
        print(f"  → {status_icon} {result['status']}{dur}")

        # Incremental save after each run
        suite_log["last_updated"] = _timestamp()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(suite_log, f, indent=2)

    suite_log["end_time"] = _timestamp()
    ok_count = sum(1 for r in suite_log["runs"] if r["status"] == "ok")
    fail_count = len(suite_log["runs"]) - ok_count
    suite_log["summary"] = {"total": len(runs), "ok": ok_count, "failed": fail_count}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(suite_log, f, indent=2)

    print()
    print(f"Suite complete: {ok_count}/{len(runs)} runs succeeded.")
    print(f"Log saved to: {output_path}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
