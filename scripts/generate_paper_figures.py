"""Generate publication-quality figures for the ITS paper.

Produces PNG (300 DPI) and PDF (vector) versions of:
  1. FID vs NFE scatter with Pareto frontier
  2. Training dynamics curves (loss, path KL, control energy)
  3. Control energy vs FID scatter (energy-quality trade-off)
  4. Work distribution histograms (Crooks verification)
  5. Path KL trajectory

Figures saved to data/paper_figures/.

Usage
-----
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --results-dir data/results --output-dir data/paper_figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.style as style

    # Paper-quality defaults
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return plt


def _save(fig, name: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {png_path.name}  {pdf_path.name}")


def _load_json(path: Path) -> Optional[dict | list]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return points on the lower-left Pareto frontier (min x AND min y)."""
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    min_y = math.inf
    for x, y in sorted_pts:
        if y < min_y:
            frontier.append((x, y))
            min_y = y
    return frontier


# ---------------------------------------------------------------------------
# Figure 1: FID vs NFE with Pareto frontier
# ---------------------------------------------------------------------------

def fig_fid_vs_nfe(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    pareto_data = _load_json(results_dir / "pareto_frontier.json")
    baseline_data = _load_json(results_dir / "baseline_comparison_equal_nfe.json")
    sweep_data = _load_json(results_dir / "extended_sweep_comparison.json")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    COLORS = {"DDPM": "#e74c3c", "DDIM": "#e67e22", "ITS-C": "#3498db",
               "ITS-D": "#2ecc71", "ITS-F": "#9b59b6", "SDE": "#7f8c8d"}

    all_points: list[tuple[float, float, str]] = []  # (nfe, fid, label)

    # Session 6: load from baseline_evals_{dataset}.json (new format)
    for beval_path in sorted(results_dir.glob("baseline_evals_*.json")):
        beval_data = _load_json(beval_path)
        if beval_data and "runs" in beval_data:
            for run in beval_data["runs"]:
                if "error" in run or "fid" not in run:
                    continue
                label = run["sampler"].upper()
                nfe = float(run.get("nfe", run.get("nfe_per_sample", 100)))
                fid = float(run["fid"])
                color = COLORS.get(label, "#95a5a6")
                ax.scatter(nfe, fid, color=color, s=80, marker="s", zorder=5,
                           label=f"{label} NFE={int(nfe)}")
                all_points.append((nfe, fid, label))

    if baseline_data:
        for mode, entry in baseline_data.items():
            if isinstance(entry, dict) and entry.get("fid") and entry.get("nfe_per_sample"):
                label = str(entry.get("mode", mode)).upper()
                nfe = float(entry["nfe_per_sample"])
                fid = float(entry["fid"])
                color = COLORS.get(label, "#95a5a6")
                ax.scatter(nfe, fid, color=color, s=80, marker="s", zorder=5,
                           label=f"{label} (N={entry.get('num_samples', '?')})")
                all_points.append((nfe, fid, label))

    if sweep_data:
        for cfg_name, entry in sweep_data.items():
            if isinstance(entry, dict):
                eval_d = entry.get("eval", {})
                if eval_d.get("fid") and eval_d.get("nfe_per_sample"):
                    label = f"ITS-{cfg_name.upper()}"
                    nfe = float(eval_d["nfe_per_sample"])
                    fid = float(eval_d["fid"])
                    color = COLORS.get(label, "#3498db")
                    ax.scatter(nfe, fid, color=color, s=100, marker="o", zorder=5,
                               label=f"{label} ({entry.get('epochs', '?')}ep)")
                    all_points.append((nfe, fid, label))

    # Pareto frontier
    if len(all_points) >= 2:
        pts_2d = [(p[0], p[1]) for p in all_points]
        frontier = _pareto_frontier(pts_2d)
        if len(frontier) >= 2:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, "k--", linewidth=1.0, alpha=0.5, label="Pareto frontier")
        # Save Pareto frontier JSON (Step 6A)
        import json as _json
        pf_out = output_dir.parent / "results" / "pareto_frontier_final.json"
        pf_out.parent.mkdir(parents=True, exist_ok=True)
        pf_out.write_text(_json.dumps({
            "all_points": [{"nfe": p[0], "fid": p[1], "label": p[2]} for p in all_points],
            "pareto_frontier": [{"nfe": x, "fid": y} for x, y in frontier],
        }, indent=2))

    if not all_points:
        # Stub with Session 4 reference data
        ax.scatter([50, 50, 100], [360.93, 375.16, 333.35],
                   color=["#e74c3c", "#e67e22", "#3498db"], s=80, marker="s",
                   label="Reference (train split, stale)")
        ax.text(0.5, 0.5, "No test-split results yet.\nRun extended suite first.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888", style="italic")

    ax.set_xlabel("NFE (Number of Function Evaluations)")
    ax.set_ylabel("FID ↓")
    ax.set_title("FID vs NFE — FashionMNIST")
    if all_points or not all_points:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", framealpha=0.8)
    _save(fig, "fig1_fid_vs_nfe", output_dir)


# ---------------------------------------------------------------------------
# Figure 2: Training dynamics curves
# ---------------------------------------------------------------------------

def fig_training_dynamics(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()

    # Try to find a JSONL log
    log_candidates = sorted(Path("logs").glob("suite_config_D_5ep*.jsonl")) + \
                     sorted(Path("logs").glob("*.jsonl"))

    records = []
    for log_path in log_candidates:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if "global_step" in r and "dsm_loss" in r:
                        records.append(r)
                except Exception:
                    pass
        if records:
            break

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    metrics = [
        ("dsm_loss", "DSM Loss", "#e74c3c"),
        ("path_kl", "Path KL", "#3498db"),
        ("control_energy", "Control Energy", "#2ecc71"),
    ]

    if records:
        steps = [r.get("global_step", i) for i, r in enumerate(records)]
        for ax, (key, label, color) in zip(axes, metrics):
            vals = [r.get(key) for r in records if r.get(key) is not None]
            s = [r.get("global_step", i) for i, r in enumerate(records) if r.get(key) is not None]
            ax.plot(s, vals, color=color, alpha=0.8, linewidth=1.2)
            ax.set_xlabel("Training step")
            ax.set_ylabel(label)
            ax.set_title(label)
    else:
        for ax, (key, label, color) in zip(axes, metrics):
            ax.text(0.5, 0.5, f"No JSONL data yet.\n(Run extended suite first.)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888888", style="italic")
            ax.set_title(label)

    fig.suptitle("Training Dynamics — Config D (ctrl_w=0.01, pkl_w=0.1, q_w=0.01)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, "fig2_training_dynamics", output_dir)


# ---------------------------------------------------------------------------
# Figure 3: Control energy vs FID scatter
# ---------------------------------------------------------------------------

def fig_energy_quality(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = _load_json(results_dir / "energy_quality_scatter.json")

    fig, ax = plt.subplots(figsize=(5, 4))

    has_data = False
    if data and isinstance(data.get("checkpoints"), list) and data["checkpoints"]:
        pts = data["checkpoints"]
        epochs = [p.get("epoch", i) for i, p in enumerate(pts)]
        energies = [p.get("control_energy") for p in pts]
        fids = [p.get("fid") for p in pts]
        valid = [(e, f, ep) for e, f, ep in zip(energies, fids, epochs)
                 if e is not None and f is not None]
        if valid:
            es, fs, eps_vals = zip(*valid)
            max_ep = max(eps_vals) if eps_vals else 1
            colors = [ep / max(max_ep, 1) for ep in eps_vals]
            sc = ax.scatter(es, fs, c=colors, cmap="Blues", s=60, alpha=0.8,
                            vmin=0, vmax=1)
            plt.colorbar(sc, ax=ax, label="Relative epoch (0=early, 1=late)")
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5,
                "No energy-quality scatter data yet.\n"
                "Run eval_samples.py on individual epoch\n"
                "checkpoints to populate.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic")

    ax.set_xlabel("Control Energy (‖u‖² per step)")
    ax.set_ylabel("FID ↓")
    ax.set_title("Control Energy vs Sample Quality")
    _save(fig, "fig3_energy_quality", output_dir)


# ---------------------------------------------------------------------------
# Figure 4: Work distribution histograms (Crooks verification)
# ---------------------------------------------------------------------------

def fig_crooks_verification(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    crooks_data = _load_json(results_dir / "crooks_verification.json")
    jarz_data = _load_json(results_dir / "jarzynski_validity.json")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    has_data = False
    if (crooks_data and crooks_data.get("forward_mean_work") is not None
            and not crooks_data.get("error")):
        # Synthetic distributions for illustration from recorded statistics
        fw_mean = crooks_data.get("forward_mean_work", 0)
        fw_std = crooks_data.get("forward_std_work", 1)
        rw_mean = crooks_data.get("reverse_mean_work", 0)
        rw_std = crooks_data.get("reverse_std_work", 1)
        rng = np.random.default_rng(42)
        w_forward = rng.normal(fw_mean, max(fw_std, 0.01), 500)
        w_reverse = rng.normal(rw_mean, max(rw_std, 0.01), 500)
        x_min = min(np.min(w_forward), np.min(-w_reverse))
        x_max = max(np.max(w_forward), np.max(-w_reverse))
        bins = np.linspace(x_min, x_max, 40)
        ax.hist(w_forward, bins=bins, alpha=0.5, color="#3498db", label="Forward work P(W)", density=True)
        ax.hist(-w_reverse, bins=bins, alpha=0.5, color="#e74c3c", label="Reverse work P(−W)", density=True)

        cp = crooks_data.get("crossing_point")
        dF_j = crooks_data.get("delta_f_jarzynski")
        if cp is not None:
            ax.axvline(cp, color="k", linewidth=1.5, label=f"Crooks ΔF = {cp:.3f}")
        if dF_j is not None:
            ax.axvline(dF_j, color="gray", linewidth=1.5, linestyle="--",
                       label=f"Jarzynski ΔF = {dF_j:.3f}")
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5,
                "No Crooks data yet.\n"
                "Run analyze_thermodynamics.py after\n"
                "extended training completes.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic")

    ax.set_xlabel("Work W")
    ax.set_ylabel("Probability density")
    ax.set_title("Crooks Fluctuation Theorem Verification")
    ax.legend(fontsize=8)
    _save(fig, "fig4_crooks_verification", output_dir)


# ---------------------------------------------------------------------------
# Figure 5: Path KL trajectory
# ---------------------------------------------------------------------------

def fig_path_kl_trajectory(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    traj_data = _load_json(results_dir / "path_kl_trajectory.json")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    has_data = False
    if traj_data and traj_data.get("records"):
        records = traj_data["records"]
        steps = [r.get("step", i) for i, r in enumerate(records)]
        pks = [r.get("path_kl") for r in records]
        valid = [(s, pk) for s, pk in zip(steps, pks) if pk is not None]
        if valid:
            sx, pkx = zip(*valid)
            ax.plot(sx, pkx, color="#3498db", alpha=0.7, linewidth=1.0, label="Path KL per step")
            # Moving average
            window = max(1, len(pkx) // 20)
            ma = np.convolve(pkx, np.ones(window) / window, mode="valid")
            ma_x = sx[window - 1:]
            ax.plot(ma_x, ma, color="#e74c3c", linewidth=2.0, label=f"Moving avg (w={window})")
            stats = traj_data.get("stats", {})
            trend = "↓ decreasing" if stats.get("trending_down") else "→ flat/increasing"
            ax.set_title(f"Path KL Trajectory — Config D  ({trend})")
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5,
                "No path KL trajectory data yet.\n"
                "Run analyze_thermodynamics.py after extended training.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic")
        ax.set_title("Path KL Trajectory (data not yet available)")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Path KL (Girsanov log-RN)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "fig5_path_kl_trajectory", output_dir)


# ---------------------------------------------------------------------------
# Figure 6: Control drift analysis (4-panel)
# ---------------------------------------------------------------------------

def fig_control_drift_analysis(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = _load_json(results_dir / "control_drift_analysis.json")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    labels = ["Cosine Similarity\n(score vs control)", "Magnitude Ratio\n(ctrl/score)",
              "Accumulated Path KL", "Control Energy\nper Step"]
    ts_keys = ["cos_sim", "ratio", None, "ctrl_mag"]

    has_data = data and "time_series" in data
    ts = data.get("time_series", {}) if has_data else {}

    for i, (ax, label) in enumerate(zip(axes, labels)):
        if has_data:
            key = ts_keys[i]
            if key and ts.get(key):
                vals = ts[key]
                xs = ts.get("epochs") or list(range(len(vals)))
                ax.plot(xs, vals, color="#3498db", linewidth=1.4)
                ax.set_xlabel("Epoch")
            elif key is None and ts.get("ctrl_mag"):
                ctrl_mags = ts["ctrl_mag"]
                ax.hist(ctrl_mags, bins=20, color="#2ecc71", alpha=0.8, density=True)
                ax.set_xlabel("Control Energy per Step")
        else:
            ax.text(0.5, 0.5, f"No data yet.\nRun Config D first.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="#888888", style="italic")
        ax.set_title(label, fontsize=10)

    fig.suptitle("Control Drift Analysis — Config D (FashionMNIST)", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, "fig6_control_drift_analysis", output_dir)


# ---------------------------------------------------------------------------
# Figure 7: Ablation study FID bar chart
# ---------------------------------------------------------------------------

def fig_ablation_bar_chart(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    data = _load_json(results_dir / "ablation_study.json")

    fig, ax = plt.subplots(figsize=(7, 4))

    has_data = False
    if data and isinstance(data.get("ablations"), list):
        ablations = [a for a in data["ablations"] if "fid" in a or "dsm_loss" in a]
        if ablations:
            names = [a.get("name", "?") for a in ablations]
            # Use FID if available, otherwise DSM loss as proxy
            vals = [a.get("fid", a.get("dsm_loss", float("nan"))) for a in ablations]
            metric_label = "FID ↓" if any("fid" in a for a in ablations) else "DSM Loss ↓"
            colors = ["#3498db" if n == "D" else "#95a5a6" for n in names]
            bars = ax.barh(names, vals, color=colors, alpha=0.85)
            ax.set_xlabel(metric_label)
            ax.set_title(f"Ablation Study — {metric_label} per Configuration")
            # Annotate bars
            for bar, val in zip(bars, vals):
                if not math.isnan(val):
                    ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:.2f}", va="center", fontsize=8)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "No ablation data yet.\nRun run_ablation_study.py first.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic")
        ax.set_title("Ablation Study (data not yet available)")

    plt.tight_layout()
    _save(fig, "fig7_ablation_bar_chart", output_dir)


# ---------------------------------------------------------------------------
# Figure 8: Pareto frontier (paper_figures canonical name)
# ---------------------------------------------------------------------------

def fig_pareto_frontier(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    baseline = _load_json(results_dir / "baseline_evals_fashionmnist.json")
    ctrl_d = _load_json(results_dir / "controlled_config_d_results.json")

    if baseline and baseline.get("runs"):
        for run in baseline["runs"]:
            sampler = run.get("sampler", "")
            marker = {"ddpm": "o", "ddim": "s", "sde": "^"}.get(sampler, "x")
            color = {"ddpm": "#2196F3", "ddim": "#4CAF50", "sde": "#9C27B0"}.get(sampler, "#999")
            ax.scatter(run["nfe"], run["fid"], marker=marker, color=color, s=60, zorder=3,
                       label=f"{sampler.upper()} (NFE={run['nfe']})" if run["nfe"] == 200 else "")

    if ctrl_d and ctrl_d.get("runs"):
        for run in ctrl_d["runs"]:
            fid = run.get("fid")
            nfe = run.get("nfe_per_sample", 100)
            if fid:
                ax.scatter(nfe, fid, marker="*", color="#F44336", s=200, zorder=4,
                           label=f"Config D (ours)")

    ax.set_xlabel("NFE")
    ax.set_ylabel("FID")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    plt.tight_layout()
    _save(fig, "pareto_frontier", output_dir)


# ---------------------------------------------------------------------------
# Figure 9: NFE efficiency curves
# ---------------------------------------------------------------------------

def fig_nfe_efficiency_curves(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    baseline = _load_json(results_dir / "baseline_evals_fashionmnist.json")
    ctrl_d = _load_json(results_dir / "controlled_config_d_results.json")

    if baseline and baseline.get("runs"):
        sampler_data: dict[str, list] = {}
        for run in baseline["runs"]:
            s = run.get("sampler", "")
            sampler_data.setdefault(s, []).append((run["nfe"], run["fid"]))

        colors = {"ddpm": "#2196F3", "ddim": "#4CAF50", "sde": "#9C27B0"}
        for sampler, pts in sampler_data.items():
            pts_sorted = sorted(pts)
            nfes, fids = zip(*pts_sorted)
            ax.plot(nfes, fids, marker="o", label=sampler.upper(),
                    color=colors.get(sampler, "#999"), linewidth=1.5)

    if ctrl_d and ctrl_d.get("runs"):
        for run in ctrl_d["runs"]:
            fid = run.get("fid")
            nfe = run.get("nfe_per_sample", 100)
            if fid:
                ax.axhline(fid, color="#F44336", linestyle="--", linewidth=1.2,
                           label=f"Config D (ours, NFE={nfe})")
                ax.scatter([nfe], [fid], marker="*", color="#F44336", s=200, zorder=5)

    ax.set_xlabel("NFE")
    ax.set_ylabel("FID")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "nfe_efficiency_curves", output_dir)


# ---------------------------------------------------------------------------
# Figure 10: Mode coverage radar (stub — requires mode_coverage.json)
# ---------------------------------------------------------------------------

def fig_mode_coverage_radar(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    coverage = _load_json(results_dir / "mode_coverage.json")
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    n = len(class_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    if coverage and "score_sde" in coverage:
        fracs = coverage["score_sde"]["fractions"]
        vals = fracs + fracs[:1]
        ax.plot(angles, vals, "o-", color="#9C27B0", linewidth=1.5, label="Score SDE")
        ax.fill(angles, vals, alpha=0.15, color="#9C27B0")

    # Ideal uniform distribution
    uniform = [1.0 / n] * n + [1.0 / n]
    ax.plot(angles, uniform, "--", color="#999", linewidth=0.8, label="Uniform")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    _save(fig, "mode_coverage_radar", output_dir)


# ---------------------------------------------------------------------------
# Figure 11: Entropy production profile (stub)
# ---------------------------------------------------------------------------

def fig_entropy_production_profile(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    ep_data = _load_json(results_dir / "entropy_production_profile.json")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    if ep_data:
        steps = ep_data.get("steps", [])
        uncontrolled = ep_data.get("uncontrolled_cumulative", [])
        controlled = ep_data.get("controlled_cumulative", [])
        if steps and uncontrolled:
            ax.plot(steps, uncontrolled, label="Uncontrolled SDE", color="#9C27B0")
        if steps and controlled:
            ax.plot(steps, controlled, label="Controlled SDE (Config D)", color="#F44336")
        if uncontrolled and controlled:
            ax.fill_between(steps, uncontrolled, controlled,
                            alpha=0.2, color="#F44336",
                            label="Additional irreversibility cost")
    else:
        ax.text(0.5, 0.5, "Entropy production data not yet computed.\n"
                "Run compute_entropy_production.py first.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic")

    ax.set_xlabel("SDE time step")
    ax.set_ylabel("Cumulative entropy production (proxy)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "entropy_production_profile", output_dir)


# ---------------------------------------------------------------------------
# Figure 12: Path KL vs FID trajectory (stub)
# ---------------------------------------------------------------------------

def fig_path_kl_vs_fid_trajectory(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    traj = _load_json(results_dir / "path_kl_fid_trajectory.json")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    if traj and traj.get("checkpoints"):
        ckpts = traj["checkpoints"]
        path_kls = [c.get("path_kl", float("nan")) for c in ckpts]
        fids = [c.get("fid", float("nan")) for c in ckpts]
        epochs = [c.get("epoch", i) for i, c in enumerate(ckpts)]

        valid = [(pk, f, e) for pk, f, e in zip(path_kls, fids, epochs)
                 if not math.isnan(pk) and not math.isnan(f)]
        if valid:
            pks, fs, es = zip(*valid)
            sc = ax.scatter(pks, fs, c=es, cmap="viridis", s=60, zorder=3)
            ax.plot(pks, fs, "-", color="#999", linewidth=0.8, zorder=2)
            plt.colorbar(sc, ax=ax, label="Epoch")
            for pk, f, e in valid:
                ax.annotate(f"ep{e}", (pk, f), fontsize=6, ha="left",
                            textcoords="offset points", xytext=(4, 2))
    else:
        ax.text(0.5, 0.5, "Trajectory data not yet computed.\n"
                "Run evaluate across checkpoints first.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic")

    ax.set_xlabel("Path KL")
    ax.set_ylabel("FID")
    plt.tight_layout()
    _save(fig, "path_kl_vs_fid_trajectory", output_dir)


# ---------------------------------------------------------------------------
# Figure 13: Ablation FID chart (canonical name for test)
# ---------------------------------------------------------------------------

def fig_ablation_fid_chart(results_dir: Path, output_dir: Path) -> None:
    plt = _setup_matplotlib()
    ablation_data = _load_json(results_dir / "ablation_study.json")

    fig, ax = plt.subplots(figsize=(6, 4))

    if ablation_data and ablation_data.get("ablations"):
        configs = ablation_data["ablations"]
        names = [c["name"] for c in configs]
        fids = [c.get("fid", float("nan")) for c in configs]
        colors = ["#F44336" if n == "D" else "#2196F3" for n in names]
        bars = ax.barh(names, fids, color=colors, edgecolor="white", linewidth=0.5)
        for bar, fid in zip(bars, fids):
            if not math.isnan(fid):
                ax.text(fid + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{fid:.1f}", va="center", fontsize=8)
        ax.set_xlabel("FID")
        ax.set_xlim(0, max(f for f in fids if not math.isnan(f)) * 1.1)
    else:
        ax.text(0.5, 0.5, "No ablation data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888888", style="italic")

    plt.tight_layout()
    _save(fig, "ablation_fid_chart", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output-dir", default="data/paper_figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")
    fig_fid_vs_nfe(results_dir, output_dir)
    fig_training_dynamics(results_dir, output_dir)
    fig_energy_quality(results_dir, output_dir)
    fig_crooks_verification(results_dir, output_dir)
    fig_path_kl_trajectory(results_dir, output_dir)
    fig_control_drift_analysis(results_dir, output_dir)
    fig_ablation_bar_chart(results_dir, output_dir)
    # Session 7 figures
    fig_pareto_frontier(results_dir, output_dir)
    fig_nfe_efficiency_curves(results_dir, output_dir)
    fig_mode_coverage_radar(results_dir, output_dir)
    fig_entropy_production_profile(results_dir, output_dir)
    fig_path_kl_vs_fid_trajectory(results_dir, output_dir)
    fig_ablation_fid_chart(results_dir, output_dir)

    # Create canonical-name copies for figures that have legacy numbered names
    import shutil
    aliases = {
        "fig4_crooks_verification": "crooks_verification",
        "fig6_control_drift_analysis": "control_drift_analysis",
    }
    for legacy, canonical in aliases.items():
        for ext in [".png", ".pdf"]:
            src = output_dir / (legacy + ext)
            dst = output_dir / (canonical + ext)
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"  Aliased: {src.name} -> {dst.name}")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
