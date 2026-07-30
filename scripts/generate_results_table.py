"""Generate LaTeX and plain-text results tables from ITS result JSON files.

Reads all JSON result files in data/results/ and produces:
  - data/results/results_table.tex  (LaTeX tabular, paper-ready)
  - data/results/results_table.txt  (plain text, human-readable)

Table columns: Sampler, Dataset, FID (N=?), IS, NFE, Ctrl Energy, Path KL, Time/sample
Formatting: best value in each column bolded, second-best underlined.
A \\midrule separates baselines from proposed methods.

Usage
-----
    python scripts/generate_results_table.py
    python scripts/generate_results_table.py --results-dir data/results --output-dir data/results
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Result file discovery and parsing
# ---------------------------------------------------------------------------

def _load_json_safe(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Canonical result-file patterns and how to parse them into rows
_RESULT_PATTERNS = {
    "baseline_comparison_equal_nfe.json": "baseline",
    "extended_sweep_comparison.json": "controlled",
    "extended_sweep_config_D.json": "controlled",
    "extended_sweep_config_C.json": "controlled",
    "extended_sweep_config_F.json": "controlled",
    "reliability_assessment.json": "reliability",
}


def _extract_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Scan results_dir for known result files and extract table rows."""
    rows: list[dict[str, Any]] = []

    # --- Session 6: multi-NFE baseline evals (run_baseline_evals.py output) ---
    for beval_path in sorted(results_dir.glob("baseline_evals_*.json")):
        data = _load_json_safe(beval_path)
        if data and "runs" in data:
            dataset = data.get("dataset", "unknown")
            for run in data["runs"]:
                if "error" in run:
                    continue
                rows.append({
                    "sampler": f"{run['sampler'].upper()} (v2)",
                    "dataset": dataset,
                    "fid": run.get("fid"),
                    "fid_n": data.get("num_samples", "?"),
                    "is_mean": run.get("inception_score_mean"),
                    "nfe": run.get("nfe", run.get("nfe_per_sample")),
                    "ctrl_energy": None,
                    "path_kl": None,
                    "time_per_sample": run.get("wall_clock_sec"),
                    "is_baseline": True,
                    "epochs": None,
                    "_note": f"Session 6 v2, NFE={run.get('nfe')}",
                })

    # --- Baseline comparison (legacy) ---
    baseline_path = results_dir / "baseline_comparison_equal_nfe.json"
    if baseline_path.is_file():
        data = _load_json_safe(baseline_path)
        if data:
            for mode, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                rows.append({
                    "sampler": entry.get("mode", mode).upper(),
                    "dataset": entry.get("dataset", "FashionMNIST"),
                    "fid": entry.get("fid"),
                    "fid_n": entry.get("num_samples", "?"),
                    "is_mean": entry.get("inception_score_mean"),
                    "nfe": entry.get("nfe_per_sample"),
                    "ctrl_energy": None,
                    "path_kl": None,
                    "time_per_sample": entry.get("wall_clock_sec"),
                    "is_baseline": True,
                    "epochs": None,
                })

    # --- Extended sweep comparison ---
    sweep_path = results_dir / "extended_sweep_comparison.json"
    if sweep_path.is_file():
        data = _load_json_safe(sweep_path)
        if data:
            for cfg_name, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                eval_data = entry.get("eval", {})
                train_data = entry.get("train", {})
                rows.append({
                    "sampler": f"ITS-{cfg_name}",
                    "dataset": entry.get("dataset", "FashionMNIST"),
                    "fid": eval_data.get("fid"),
                    "fid_n": eval_data.get("num_samples", 2048),
                    "is_mean": eval_data.get("inception_score_mean"),
                    "nfe": eval_data.get("nfe_per_sample"),
                    "ctrl_energy": train_data.get("control_energy"),
                    "path_kl": train_data.get("path_kl"),
                    "time_per_sample": eval_data.get("wall_clock_sec"),
                    "is_baseline": False,
                    "epochs": entry.get("epochs"),
                })

    # --- Individual config files ---
    for fname in ["extended_sweep_config_C.json", "extended_sweep_config_D.json", "extended_sweep_config_F.json"]:
        p = results_dir / fname
        if p.is_file():
            data = _load_json_safe(p)
            if data and isinstance(data, dict) and "fid" in data:
                cfg_label = fname.replace("extended_sweep_", "").replace(".json", "")
                rows.append({
                    "sampler": f"ITS-{cfg_label}",
                    "dataset": data.get("dataset", "FashionMNIST"),
                    "fid": data.get("fid"),
                    "fid_n": data.get("num_samples", 2048),
                    "is_mean": data.get("inception_score_mean"),
                    "nfe": data.get("nfe_per_sample"),
                    "ctrl_energy": data.get("control_energy"),
                    "path_kl": data.get("path_kl"),
                    "time_per_sample": data.get("wall_clock_sec"),
                    "is_baseline": False,
                    "epochs": data.get("epochs"),
                })

    # --- Session 6: controlled Config D results (train_controlled_config_d.py output) ---
    config_d_path = results_dir / "controlled_config_d_results.json"
    if config_d_path.is_file():
        data = _load_json_safe(config_d_path)
        if data and "runs" in data:
            dsm_vals = [r.get("dsm_loss") for r in data["runs"] if r.get("dsm_loss") is not None]
            fid_vals = [r.get("fid") for r in data["runs"] if r.get("fid") is not None]
            fid_mean = sum(fid_vals) / len(fid_vals) if fid_vals else None
            dsm_mean = sum(dsm_vals) / len(dsm_vals) if dsm_vals else None
            rows.append({
                "sampler": "ITS-Config-D (controlled)",
                "dataset": "FashionMNIST",
                "fid": fid_mean,
                "fid_n": 5000,
                "is_mean": None,
                "nfe": 100,
                "ctrl_energy": data["runs"][0].get("control_energy") if data["runs"] else None,
                "path_kl": data["runs"][0].get("path_kl") if data["runs"] else None,
                "time_per_sample": None,
                "is_baseline": False,
                "epochs": data.get("epochs"),
                "_note": f"Config D, {len(data['runs'])} seeds, mean FID",
            })

    # --- Session 8: controlled Config D 3-seed reliability ---
    multiseed_path = results_dir / "controlled_config_d_multiseed.json"
    if multiseed_path.is_file():
        data = _load_json_safe(multiseed_path)
        if data and "ci_95" in data:
            ci = data["ci_95"]
            rows.append({
                "sampler": "ITS-Config-D (3-seed CI)",
                "dataset": "FashionMNIST",
                "fid": ci.get("mean"),
                "fid_ci_lower": ci.get("lower"),
                "fid_ci_upper": ci.get("upper"),
                "fid_std": ci.get("std"),
                "fid_n": data.get("num_samples", 2048),
                "is_mean": None,
                "nfe": data.get("nfe", 100),
                "ctrl_energy": None,
                "path_kl": None,
                "time_per_sample": None,
                "is_baseline": False,
                "epochs": data.get("epochs", 60),
                "_note": f"Config D, 3 seeds, 95% CI [{ci.get('lower', 0):.2f}, {ci.get('upper', 0):.2f}]",
            })

    # --- Session 9: ConvControl results ---
    conv_path = results_dir / "controlled_conv_results.json"
    if conv_path.is_file():
        data = _load_json_safe(conv_path)
        if data and "runs" in data:
            fid_vals = [r.get("fid") for r in data["runs"] if r.get("fid") is not None]
            fid_mean = sum(fid_vals) / len(fid_vals) if fid_vals else None
            rows.append({
                "sampler": "ITS-ConvControl",
                "dataset": "FashionMNIST",
                "fid": fid_mean,
                "fid_n": data["runs"][0].get("num_samples_eval", 2048) if data["runs"] else 2048,
                "is_mean": None,
                "nfe": 100,
                "ctrl_energy": None,
                "path_kl": None,
                "time_per_sample": None,
                "is_baseline": False,
                "epochs": data.get("epochs"),
                "_note": f"ConvControl {data.get('epochs')}ep, {len(data['runs'])} seed(s)",
            })

    # --- Session 9: ConvControl multiseed reliability ---
    conv_ms_path = results_dir / "controlled_conv_multiseed.json"
    if conv_ms_path.is_file():
        data = _load_json_safe(conv_ms_path)
        if data and "ci_95" in data:
            ci = data["ci_95"]
            rows.append({
                "sampler": "ITS-ConvControl (3-seed CI)",
                "dataset": "FashionMNIST",
                "fid": ci.get("mean"),
                "fid_ci_lower": ci.get("lower"),
                "fid_ci_upper": ci.get("upper"),
                "fid_std": ci.get("std"),
                "fid_n": data.get("num_samples", 2048),
                "is_mean": None,
                "nfe": data.get("nfe", 100),
                "ctrl_energy": None,
                "path_kl": None,
                "time_per_sample": None,
                "is_baseline": False,
                "epochs": data.get("epochs", 50),
                "_note": f"ConvControl, 3 seeds, 95% CI [{ci.get('lower', 0):.2f}, {ci.get('upper', 0):.2f}]",
            })

    # --- Session 6: ablation study results (run_ablation_study.py output) ---
    ablation_path = results_dir / "ablation_study.json"
    if ablation_path.is_file():
        data = _load_json_safe(ablation_path)
        if data and "ablations" in data:
            for abl in data["ablations"]:
                if abl.get("status") != "ok":
                    continue
                rows.append({
                    "sampler": f"ITS-Ablation-{abl['name']}",
                    "dataset": "FashionMNIST",
                    "fid": abl.get("fid"),
                    "fid_n": data.get("num_samples", 2048),
                    "is_mean": abl.get("inception_score_mean"),
                    "nfe": abl.get("nfe_per_sample", 100),
                    "ctrl_energy": abl.get("control_energy"),
                    "path_kl": abl.get("path_kl"),
                    "time_per_sample": None,
                    "is_baseline": False,
                    "epochs": data.get("epochs"),
                    "_note": f"Ablation {abl['name']}, {data.get('epochs')}ep",
                })

    # Fallback: if no results exist, add stub rows from Session 4 (train-split FIDs, stale)
    if not rows:
        rows = [
            {
                "sampler": "ITS-Ctrl-SDE (stale)", "dataset": "FashionMNIST",
                "fid": 333.35, "fid_n": 256, "is_mean": 1.24, "nfe": 100,
                "ctrl_energy": None, "path_kl": None, "time_per_sample": None,
                "is_baseline": False, "epochs": 3,
                "_note": "Session 1-3 result, TRAIN split (incorrect), for reference only",
            },
            {
                "sampler": "DDPM", "dataset": "FashionMNIST",
                "fid": 360.93, "fid_n": 256, "is_mean": 1.50, "nfe": 50,
                "ctrl_energy": None, "path_kl": None, "time_per_sample": None,
                "is_baseline": True, "epochs": None,
                "_note": "Session 1-3 result, TRAIN split (incorrect), for reference only",
            },
            {
                "sampler": "DDIM", "dataset": "FashionMNIST",
                "fid": 375.16, "fid_n": 256, "is_mean": 1.34, "nfe": 50,
                "ctrl_energy": None, "path_kl": None, "time_per_sample": None,
                "is_baseline": True, "epochs": None,
                "_note": "Session 1-3 result, TRAIN split (incorrect), for reference only",
            },
        ]

    # Separate baselines and proposed
    baselines = [r for r in rows if r.get("is_baseline")]
    proposed = [r for r in rows if not r.get("is_baseline")]
    return baselines, proposed


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float], decimals: int = 2, bold: bool = False, underline: bool = False) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        s = "—"
    else:
        s = f"{val:.{decimals}f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    elif underline:
        s = f"\\underline{{{s}}}"
    return s


def _fmt_plain(val: Optional[float], decimals: int = 2, marker: str = "") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "  —  " + " " * max(0, decimals - 1)
    return f"{val:.{decimals}f}{marker}"


def _rank_column(rows: list[dict], key: str, lower_is_better: bool = True) -> dict[int, int]:
    """Return {row_index: rank} where rank 1 = best."""
    vals = [(i, r.get(key)) for i, r in enumerate(rows) if r.get(key) is not None]
    sorted_vals = sorted(vals, key=lambda x: x[1], reverse=not lower_is_better)
    return {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_vals)}


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_latex(baselines: list[dict], proposed: list[dict]) -> str:
    all_rows = baselines + proposed
    n_base = len(baselines)

    # Compute rankings across all rows
    fid_rank = _rank_column(all_rows, "fid", lower_is_better=True)
    nfe_rank = _rank_column(all_rows, "nfe", lower_is_better=True)
    is_rank = _rank_column(all_rows, "is_mean", lower_is_better=False)

    lines = [
        r"% Auto-generated by scripts/generate_results_table.py",
        r"% ITS: Information-Theoretic Sampler — Results Table",
        r"% FID values marked (stale) were computed against training split (incorrect).",
        r"% All other FID values use the test split.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{%",
        r"    ITS evaluation results on FashionMNIST.",
        r"    FID lower is better; IS higher is better.",
        r"    $N$ = number of generated samples used for FID estimation.",
        r"    \textbf{Bold}: best per column; \underline{underline}: second-best.",
        r"    Results marked $\dagger$ use the training split and are included for historical",
        r"    reference only; all other results use the test split.",
        r"  }",
        r"  \label{tab:main_results}",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        r"    Sampler & Dataset & FID ($N$) & IS & NFE & Ctrl$\,E$ & Path KL \\",
        r"    \midrule",
    ]

    for i, row in enumerate(all_rows):
        if i == n_base and n_base > 0:
            lines.append(r"    \midrule  % baselines above, proposed below")

        sampler = row["sampler"].replace("_", r"\_")
        dataset = row.get("dataset", "FashionMNIST")
        fid_n = row.get("fid_n", "?")
        note = row.get("_note", "")
        stale = "stale" in note.lower() or "train split" in note.lower()
        dagger = r"$^\dagger$" if stale else ""

        fid_str = _fmt(row.get("fid"), 2,
                        bold=(fid_rank.get(i) == 1),
                        underline=(fid_rank.get(i) == 2))
        is_str = _fmt(row.get("is_mean"), 2,
                       bold=(is_rank.get(i) == 1),
                       underline=(is_rank.get(i) == 2))
        nfe_str = _fmt(row.get("nfe"), 0,
                        bold=(nfe_rank.get(i) == 1),
                        underline=(nfe_rank.get(i) == 2))
        ce_str = _fmt(row.get("ctrl_energy"), 4)
        pk_str = _fmt(row.get("path_kl"), 3)

        lines.append(
            f"    {sampler}{dagger} & {dataset} & "
            f"{fid_str} ($N$={fid_n}) & {is_str} & {nfe_str} & "
            f"{ce_str} & {pk_str} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_plain(baselines: list[dict], proposed: list[dict]) -> str:
    all_rows = baselines + proposed
    n_base = len(baselines)
    fid_rank = _rank_column(all_rows, "fid", lower_is_better=True)
    nfe_rank = _rank_column(all_rows, "nfe", lower_is_better=True)

    header = (
        f"{'Sampler':<22} {'Dataset':<14} {'FID':>8} {'N':>6} {'IS':>6} "
        f"{'NFE':>5} {'Ctrl-E':>8} {'Path-KL':>9} {'Epochs':>7}"
    )
    sep = "-" * len(header)
    lines = [
        "ITS Results Table",
        "=" * len(header),
        "NOTE: Rows marked [stale] used training split for FID — included for reference only.",
        "All other rows use the test split (train=False).",
        "",
        header,
        sep,
    ]
    for i, row in enumerate(all_rows):
        if i == n_base and n_base > 0:
            lines.append(sep + " ← baselines above, proposed below")

        fid_val = row.get("fid")
        fid_s = f"{fid_val:.2f}" if fid_val is not None else "—"
        fid_marker = " *" if fid_rank.get(i) == 1 else (" +" if fid_rank.get(i) == 2 else "  ")

        is_val = row.get("is_mean")
        nfe_val = row.get("nfe")
        ce_val = row.get("ctrl_energy")
        pk_val = row.get("path_kl")
        ep_val = row.get("epochs")
        note = row.get("_note", "")
        stale = "stale" in note.lower() or "train split" in note.lower()
        stale_s = " [stale]" if stale else ""

        lines.append(
            f"{row['sampler'][:22]:<22} {row.get('dataset','FashionMNIST')[:14]:<14} "
            f"{fid_s:>8}{fid_marker}"
            f"{str(row.get('fid_n','?')):>6} "
            f"{f'{is_val:.2f}' if is_val else '—':>6} "
            f"{f'{nfe_val:.0f}' if nfe_val else '—':>5} "
            f"{f'{ce_val:.4f}' if ce_val else '—':>8} "
            f"{f'{pk_val:.3f}' if pk_val else '—':>9} "
            f"{str(ep_val) if ep_val else '—':>7}"
            f"{stale_s}"
        )

    lines += [
        sep,
        "* = best in column   + = second-best",
        "",
        "LaTeX version: data/results/results_table.tex",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines, proposed = _extract_rows(results_dir)
    print(f"Found {len(baselines)} baseline rows, {len(proposed)} proposed rows.")

    latex_str = generate_latex(baselines, proposed)
    plain_str = generate_plain(baselines, proposed)

    tex_path = output_dir / "results_table.tex"
    txt_path = output_dir / "results_table.txt"
    tex_path.write_text(latex_str, encoding="utf-8")
    txt_path.write_text(plain_str, encoding="utf-8")
    print(f"LaTeX table: {tex_path}")
    print(f"Plain text:  {txt_path}")


if __name__ == "__main__":
    main()
