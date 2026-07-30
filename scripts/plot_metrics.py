from __future__ import annotations

"""
Plot metrics and regenerate figures from collected CSV/JSON metrics.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List

from its.analysis.metrics import plot_histogram, plot_scatter, plot_time_series


def _load_records(path: Path) -> List[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    records: List[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {}
            for k, v in row.items():
                try:
                    rec[k] = float(v)
                except Exception:
                    rec[k] = v
            records.append(rec)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ITS metrics from CSV/JSON files.")
    parser.add_argument("--metrics", required=True, help="Metrics file (CSV or JSON).")
    parser.add_argument("--type", required=True, choices=["toy", "score", "controlled", "eval"], help="Experiment type.")
    parser.add_argument("--output-dir", required=True, help="Directory to save figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return
    records = _load_records(metrics_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "toy":
        # Occupancy balance over steps
        plot_time_series(records, x_key="step", y_keys=["bal", "occupancy_balance"], title="Toy Occupancy Balance", out_path=out_dir / "fig_toy_occupancy_balance.png")
        # Quality vs control over steps
        plot_scatter(records, x_key="control", y_key="quality", title="Toy Quality vs Control", out_path=out_dir / "fig_toy_quality_vs_control.png")
        # Drift alignment / entropy / variance
        plot_time_series(records, x_key="step", y_keys=["entropy", "spread"], title="Toy Entropy/Spread", out_path=out_dir / "fig_toy_entropy_spread.png")
    elif args.type == "score":
        # Loss vs step/epoch
        plot_time_series(records, x_key="step", y_keys=["loss"], title="Score Training Loss vs Step", out_path=out_dir / "fig_score_loss_step.png")
        plot_time_series(records, x_key="epoch", y_keys=["avg_loss"], title="Score Training Avg Loss vs Epoch", out_path=out_dir / "fig_score_loss_epoch.png")
    elif args.type == "controlled":
        # Controlled image training: DSM, control energy, path_kl, jarzynski vs step
        plot_time_series(records, x_key="step", y_keys=["loss", "dsm", "control", "path_kl", "jar"], title="Controlled Training Metrics vs Step", out_path=out_dir / "fig_controlled_metrics.png")
        plot_scatter(records, x_key="control", y_key="path_kl", title="Control Energy vs Path KL", out_path=out_dir / "fig_control_vs_pathkl.png")
        plot_histogram([r["jar"] for r in records if "jar" in r], title="Jarzynski Histogram", out_path=out_dir / "fig_controlled_jar_hist.png")
    elif args.type == "eval":
        # Evaluation: FID/IS vs runs; also trade-offs
        plot_scatter(records, x_key="fid", y_key="inception_score_mean", title="FID vs Inception Score", out_path=out_dir / "fig_eval_fid_vs_is.png")
        if any("control_energy_total" in r for r in records):
            plot_scatter(records, x_key="fid", y_key="control_energy_total", title="FID vs Control Energy", out_path=out_dir / "fig_eval_fid_vs_control_energy.png")
        if any("path_kl" in r for r in records):
            plot_scatter(records, x_key="fid", y_key="path_kl", title="FID vs Path KL", out_path=out_dir / "fig_eval_fid_vs_pathkl.png")

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
