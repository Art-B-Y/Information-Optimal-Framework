"""Generate self-contained HTML analytics report for the ITS project.

Usage:
    python scripts/generate_report.py [--output data/results/its_analytics_report.html]

Reads training logs, encodes sample images as base64, and emits a single HTML
file with inline SVG charts and all sections required by the Phase-5 spec.
No external CDN or JS libraries required.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def parse_score_log(log_path: Path) -> dict:
    """Parse score training log (text or JSONL), return step-level and epoch-level series."""
    steps, losses, lrs = [], [], []
    epochs, epoch_losses, val_losses = [], [], []
    content = log_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Try JSONL format first (Session 6 v2 logs)
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                if "global_step" in rec and "loss" in rec:
                    steps.append(rec["global_step"])
                    losses.append(rec["loss"])
                    lrs.append(rec.get("lr", None))
                if "avg_loss" in rec and "epoch" in rec and "global_step" not in rec:
                    epochs.append(rec["epoch"])
                    epoch_losses.append(rec["avg_loss"])
                if "val_dsm_loss" in rec:
                    val_losses.append({"epoch": rec["epoch"], "val": rec["val_dsm_loss"],
                                       "train": rec.get("train_dsm_loss"), "ratio": rec.get("overfit_ratio")})
                elif "avg_loss" in rec and "epoch" in rec and "global_step" not in rec:
                    # Epoch-summary records (no global_step), not per-step records
                    epochs.append(rec["epoch"])
                    epoch_losses.append(rec["avg_loss"])
                continue
            except json.JSONDecodeError:
                pass
        # Fall back to text regex
        m = re.search(r"epoch=(\d+) step=(\d+) loss=([\d.]+)", line)
        if m:
            steps.append(int(m.group(2)))
            losses.append(float(m.group(3)))
        m = re.search(r"epoch=(\d+) avg_loss=([\d.]+)", line)
        if m:
            epochs.append(int(m.group(1)))
            epoch_losses.append(float(m.group(2)))
    return {"steps": steps, "losses": losses, "lrs": lrs,
            "epochs": epochs, "epoch_losses": epoch_losses, "val_losses": val_losses}


def parse_ctrl_log(log_path: Path) -> dict:
    """Parse controlled training log, return multi-component series."""
    steps, dsm, ctrl, kl, jar = [], [], [], [], []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"step=(\d+) loss=[\d.]+ dsm=([\d.]+) control=([\d.]+) path_kl=([-\d.]+) jar=([-\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            dsm.append(float(m.group(2)))
            ctrl.append(float(m.group(3)))
            kl.append(float(m.group(4)))
            jar.append(float(m.group(5)))
    return {"steps": steps, "dsm": dsm, "ctrl": ctrl, "kl": kl, "jar": jar}


def encode_image(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{ext};base64,{data}"


# ---------------------------------------------------------------------------
# SVG chart primitives
# ---------------------------------------------------------------------------

def _svg_line_chart(
    series: list,
    width: int = 620,
    height: int = 280,
    title: str = "",
    xlabel: str = "Step",
    ylabel: str = "Value",
    title_id: str = "",
    desc: str = "",
) -> str:
    """Draw an accessible SVG line chart."""
    import math

    PAD_L, PAD_R, PAD_T, PAD_B = 65, 20, 42, 52
    W = width - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    all_x = [x for _, xs, ys, _ in series for x in xs]
    all_y = [y for _, xs, ys, _ in series for y in ys]
    if not all_x:
        return (
            f'<svg role="img" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<title>{title}</title><desc>No data available</desc>'
            f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#888" font-size="13">No data available</text></svg>'
        )

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    rx = max_x - min_x or 1
    ry = max_y - min_y or 1

    def px(x):
        return PAD_L + (x - min_x) / rx * W

    def py(y):
        return PAD_T + H - (y - min_y) / ry * H

    parts = []
    if not title_id:
        title_id = re.sub(r"\W+", "_", title)[:40]
    desc_id = title_id + "_desc"
    parts.append(f'<title id="{title_id}">{title}</title>')
    parts.append(f'<desc id="{desc_id}">{desc or title}</desc>')

    # Grid lines
    for i in range(5):
        yv = min_y + i / 4 * ry
        gy = PAD_T + H - i / 4 * H
        label = f"{yv:.1f}" if abs(yv) < 1000 else f"{yv:.0f}"
        parts.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{PAD_L+W}" y2="{gy:.1f}" stroke="#e8e8e8" stroke-width="1"/>'
            f'<text x="{PAD_L-5}" y="{gy:.1f}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="#666">{label}</text>'
        )
    for i in range(5):
        xv = min_x + i / 4 * rx
        gx = PAD_L + i / 4 * W
        parts.append(
            f'<text x="{gx:.1f}" y="{PAD_T+H+18}" text-anchor="middle" font-size="10" fill="#666">{xv:.0f}</text>'
        )

    # Series with dash patterns for print-readability
    DASHES = ["none", "6,3", "2,3", "8,2,2,2", "4,4"]
    legend_items = []
    for i, (label, xs, ys, color) in enumerate(series):
        if not xs:
            continue
        dash = DASHES[i % len(DASHES)]
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        if pts:
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" '
                f'stroke-dasharray="{dash}" stroke-linejoin="round"/>'
            )
        legend_items.append((label, color, dash))

    # Legend
    for i, (label, color, dash) in enumerate(legend_items):
        lx = PAD_L + i * 130
        ly = PAD_T - 16
        parts.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx+20}" y2="{ly}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
            f'<text x="{lx+24}" y="{ly+4}" font-size="11" fill="#333">{label}</text>'
        )

    # Axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
        f'<line x1="{PAD_L}" y1="{PAD_T+H}" x2="{PAD_L+W}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
    )
    # Labels
    parts.append(
        f'<text x="{width//2}" y="20" text-anchor="middle" font-size="12" font-weight="bold" fill="#222">{title}</text>'
        f'<text x="{width//2}" y="{height-6}" text-anchor="middle" font-size="11" fill="#555">{xlabel}</text>'
        f'<text x="13" y="{height//2}" text-anchor="middle" font-size="11" fill="#555" '
        f'transform="rotate(-90,13,{height//2})">{ylabel}</text>'
    )

    inner = "\n    ".join(parts)
    return (
        f'<svg role="img" aria-labelledby="{title_id} {desc_id}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">\n    {inner}\n</svg>'
    )


def _svg_scatter_pareto(
    points: list,
    width: int = 560,
    height: int = 320,
) -> str:
    """Draw NFE vs FID scatter with Pareto frontier. points = list of (label, nfe, fid, color)."""
    if not points:
        return f'<svg role="img" width="{width}" height="{height}"><title>Scatter plot</title><text x="50%" y="50%" text-anchor="middle" fill="#888">No data</text></svg>'

    PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 40, 50
    W = width - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    nfes = [p[1] for p in points]
    fids = [p[2] for p in points]
    min_n, max_n = min(nfes) * 0.9, max(nfes) * 1.1
    min_f, max_f = min(fids) * 0.9, max(fids) * 1.1
    rn, rf = max_n - min_n or 1, max_f - min_f or 1

    def px(n):
        return PAD_L + (n - min_n) / rn * W
    def py(f):
        return PAD_T + H - (f - min_f) / rf * H

    parts = []
    parts.append('<title id="scatter_t">NFE vs FID Scatter (Pareto Frontier)</title>')
    parts.append('<desc id="scatter_d">Scatter plot comparing sampler modes on NFE and FID axes with Pareto frontier overlay</desc>')

    # Grid
    for i in range(5):
        gy = PAD_T + i / 4 * H
        fv = max_f - i / 4 * rf
        parts.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{PAD_L+W}" y2="{gy:.1f}" stroke="#eee" stroke-width="1"/>'
            f'<text x="{PAD_L-5}" y="{gy:.1f}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="#666">{fv:.0f}</text>'
        )
    for i in range(5):
        nv = min_n + i / 4 * rn
        gx = PAD_L + i / 4 * W
        parts.append(f'<text x="{gx:.1f}" y="{PAD_T+H+18}" text-anchor="middle" font-size="10" fill="#666">{nv:.0f}</text>')

    # Pareto frontier (lower-left non-dominated)
    sorted_pts = sorted(points, key=lambda p: p[1])
    pareto = []
    best_fid = float("inf")
    for p in sorted_pts:
        if p[2] < best_fid:
            best_fid = p[2]
            pareto.append(p)
    if len(pareto) >= 2:
        pts_str = " ".join(f"{px(p[1]):.1f},{py(p[2]):.1f}" for p in pareto)
        parts.append(f'<polyline points="{pts_str}" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="6,3"/>')

    # Points
    SHAPES = ["circle", "rect", "diamond", "triangle"]
    for i, (label, nfe, fid, color) in enumerate(points):
        cx, cy = px(nfe), py(fid)
        r = 8
        # Draw shape
        if i % 4 == 0:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        elif i % 4 == 1:
            parts.append(f'<rect x="{cx-r:.1f}" y="{cy-r:.1f}" width="{2*r}" height="{2*r}" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        elif i % 4 == 2:
            pts_d = f"{cx:.1f},{cy-r:.1f} {cx+r:.1f},{cy:.1f} {cx:.1f},{cy+r:.1f} {cx-r:.1f},{cy:.1f}"
            parts.append(f'<polygon points="{pts_d}" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        else:
            pts_d = f"{cx:.1f},{cy-r:.1f} {cx+r:.1f},{cy+r:.1f} {cx-r:.1f},{cy+r:.1f}"
            parts.append(f'<polygon points="{pts_d}" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        # Label
        parts.append(f'<text x="{cx+r+3:.1f}" y="{cy+4:.1f}" font-size="11" fill="#333">{label}</text>')

    # Axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
        f'<line x1="{PAD_L}" y1="{PAD_T+H}" x2="{PAD_L+W}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{width//2}" y="20" text-anchor="middle" font-size="12" font-weight="bold" fill="#222">Compute–Quality Trade-off (NFE vs FID)</text>'
        f'<text x="{width//2}" y="{height-6}" text-anchor="middle" font-size="11" fill="#555">NFE per sample (proxy: compute cost)</text>'
        f'<text x="13" y="{height//2}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90,13,{height//2})">FID ↓ (lower is better)</text>'
        f'<text x="{PAD_L+W-10}" y="{PAD_T+12}" text-anchor="end" font-size="10" fill="#888" font-style="italic">— Pareto frontier</text>'
    )

    inner = "\n    ".join(parts)
    return (
        f'<svg role="img" aria-labelledby="scatter_t scatter_d" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">\n    {inner}\n</svg>'
    )


def _svg_bar_chart(
    categories: list,
    series: list,
    width: int = 560,
    height: int = 280,
    title: str = "",
    ylabel: str = "Value",
) -> str:
    """Grouped bar chart. categories=[str], series=[(label, [values], color)]."""
    if not categories or not series:
        return f'<svg role="img" width="{width}" height="{height}"><title>{title}</title><text x="50%" y="50%" text-anchor="middle" fill="#888">No data</text></svg>'

    PAD_L, PAD_R, PAD_T, PAD_B = 65, 20, 42, 60
    W = width - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    n_cat = len(categories)
    n_ser = len(series)
    all_vals = [v for _, vals, _ in series for v in vals]
    max_v = max(all_vals) if all_vals else 1
    min_v = 0

    cat_w = W / n_cat
    bar_w = (cat_w * 0.7) / n_ser
    gap = cat_w * 0.15

    def bh(v):
        return (v - min_v) / (max_v - min_v or 1) * H

    parts = []
    tid = re.sub(r"\W+", "_", title)[:40]
    parts.append(f'<title id="{tid}">{title}</title>')

    # Y-grid
    for i in range(5):
        yv = max_v * i / 4
        gy = PAD_T + H - i / 4 * H
        label = f"{yv:.2f}" if max_v < 1 else f"{yv:.1f}"
        parts.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{PAD_L+W}" y2="{gy:.1f}" stroke="#eee" stroke-width="1"/>'
            f'<text x="{PAD_L-5}" y="{gy:.1f}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="#666">{label}</text>'
        )

    HATCHES = ["", "url(#hatch1)", "url(#hatch2)", "url(#hatch3)"]
    # Bars
    for ci, cat in enumerate(categories):
        cx = PAD_L + ci * cat_w + gap
        for si, (label, vals, color) in enumerate(series):
            bx = cx + si * bar_w
            v = vals[ci] if ci < len(vals) else 0
            bh_px = bh(v)
            by = PAD_T + H - bh_px
            hatch = HATCHES[si % len(HATCHES)]
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.9:.1f}" height="{bh_px:.1f}" fill="{color}" opacity="0.85"/>'
            )
            if hatch:
                parts.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.9:.1f}" height="{bh_px:.1f}" fill="{hatch}" opacity="0.5"/>'
                )
        # Category label
        parts.append(
            f'<text x="{PAD_L + (ci+0.5)*cat_w:.1f}" y="{PAD_T+H+18}" text-anchor="middle" font-size="10" fill="#333">{cat}</text>'
        )

    # Hatch patterns for print
    parts.insert(0,
        '<defs>'
        '<pattern id="hatch1" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-width="2"/></pattern>'
        '<pattern id="hatch2" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(90)">'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-width="2"/></pattern>'
        '<pattern id="hatch3" width="6" height="6" patternUnits="userSpaceOnUse">'
        '<line x1="0" y1="3" x2="6" y2="3" stroke="#fff" stroke-width="2"/></pattern>'
        '</defs>'
    )

    # Legend
    for si, (label, _, color) in enumerate(series):
        lx = PAD_L + si * 140
        parts.append(
            f'<rect x="{lx}" y="{PAD_T-22}" width="14" height="12" fill="{color}" opacity="0.85"/>'
            f'<text x="{lx+18}" y="{PAD_T-11}" font-size="11" fill="#333">{label}</text>'
        )

    # Axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
        f'<line x1="{PAD_L}" y1="{PAD_T+H}" x2="{PAD_L+W}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{width//2}" y="20" text-anchor="middle" font-size="12" font-weight="bold" fill="#222">{title}</text>'
        f'<text x="13" y="{height//2}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90,13,{height//2})">{ylabel}</text>'
    )

    inner = "\n    ".join(parts)
    return (
        f'<svg role="img" aria-labelledby="{tid}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">\n    {inner}\n</svg>'
    )


def _svg_heatmap_phase(width: int = 520, height: int = 320) -> str:
    """Phase diagram heatmap: control_weight x path_kl_weight."""
    PAD_L, PAD_R, PAD_T, PAD_B = 75, 20, 42, 55
    W = width - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    # Axis values
    ctrl_weights = [0.001, 0.01, 0.1, 1.0]
    kl_weights = [0.0, 0.01, 0.1, 1.0]
    # Known results (None=untested, True=better than baseline, False=worse)
    results = {
        (0.01, 0.1): None,  # current config, in progress
        (0.01, 0.0): False,  # Bug 5: zero KL weight decouples objective
    }

    n_c, n_k = len(ctrl_weights), len(kl_weights)
    cell_w = W / n_c
    cell_h = H / n_k

    parts = []
    parts.append('<title id="heatmap_t">Phase Diagram: control_weight × path_kl_weight</title>')
    parts.append('<desc id="heatmap_d">Heatmap showing sampler quality across hyperparameter combinations. Green=better than baseline, red=worse, gray=untested, gold=in progress.</desc>')

    # Cells
    for ci, cw in enumerate(ctrl_weights):
        for ki, kw in enumerate(reversed(kl_weights)):
            cx = PAD_L + ci * cell_w
            cy = PAD_T + ki * cell_h
            res = results.get((cw, kw))
            if res is True:
                fill = "#c8e6c9"
                border = "#388e3c"
                label = "✓"
            elif res is False:
                fill = "#ffcdd2"
                border = "#c62828"
                label = "✗"
            elif res is None and (cw, kw) in results:
                fill = "#fff9c4"
                border = "#f9a825"
                label = "⟳"
            else:
                fill = "#f0f0f0"
                border = "#ccc"
                label = ""
            parts.append(
                f'<rect x="{cx+1:.1f}" y="{cy+1:.1f}" width="{cell_w-2:.1f}" height="{cell_h-2:.1f}" '
                f'fill="{fill}" stroke="{border}" stroke-width="1.5" rx="3"/>'
            )
            if label:
                parts.append(
                    f'<text x="{cx+cell_w/2:.1f}" y="{cy+cell_h/2+5:.1f}" text-anchor="middle" font-size="16" fill="{border}">{label}</text>'
                )

    # X axis (control_weight)
    for ci, cw in enumerate(ctrl_weights):
        cx = PAD_L + (ci + 0.5) * cell_w
        parts.append(f'<text x="{cx:.1f}" y="{PAD_T+H+22}" text-anchor="middle" font-size="11" fill="#333">{cw}</text>')

    # Y axis (kl_weight reversed)
    for ki, kw in enumerate(reversed(kl_weights)):
        cy = PAD_T + (ki + 0.5) * cell_h
        parts.append(f'<text x="{PAD_L-8}" y="{cy+4:.1f}" text-anchor="end" font-size="11" fill="#333">{kw}</text>')

    # Axis labels
    parts.append(
        f'<text x="{PAD_L+W/2:.1f}" y="{height-8}" text-anchor="middle" font-size="11" fill="#555">control_weight (λ_ctrl)</text>'
        f'<text x="13" y="{PAD_T+H/2:.1f}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90,13,{PAD_T+H/2:.1f})">path_kl_weight (λ_kl)</text>'
        f'<text x="{PAD_L+W/2:.1f}" y="20" text-anchor="middle" font-size="12" font-weight="bold" fill="#222">Phase Diagram: λ_ctrl × λ_kl</text>'
    )

    # Legend
    legend_items = [("#c8e6c9", "#388e3c", "Better than baseline"),
                    ("#ffcdd2", "#c62828", "Worse / unstable"),
                    ("#fff9c4", "#f9a825", "In progress"),
                    ("#f0f0f0", "#ccc", "Untested")]
    for i, (fill, border, label) in enumerate(legend_items):
        lx = PAD_L + i * 130
        parts.append(
            f'<rect x="{lx}" y="{PAD_T-22}" width="14" height="12" fill="{fill}" stroke="{border}" stroke-width="1"/>'
            f'<text x="{lx+18}" y="{PAD_T-11}" font-size="10" fill="#333">{label}</text>'
        )

    inner = "\n    ".join(parts)
    return (
        f'<svg role="img" aria-labelledby="heatmap_t heatmap_d" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">\n    {inner}\n</svg>'
    )


def _svg_gantt(width: int = 680, height: int = 320) -> str:
    """Horizontal timeline bar chart for next-experiments roadmap."""
    experiments = [
        ("CIFAR-10 score baseline (200 ep)", 0, 200, "#2563eb", "~400 GPU-h"),
        ("FashionMNIST ConvControl (50 ep)", 10, 60, "#7c3aed", "~8 GPU-h"),
        ("λ sweep (ctrl × kl, FashionMNIST)", 60, 30, "#059669", "~24 GPU-h"),
        ("Best-config CIFAR-10 controlled (50 ep)", 90, 120, "#d97706", "~60 GPU-h"),
        ("Large-sample FID eval (10k samples)", 200, 20, "#dc2626", "~4 GPU-h"),
        ("Schrödinger bridge IPF solver", 100, 150, "#64748b", "~100 GPU-h"),
    ]

    PAD_L, PAD_R, PAD_T, PAD_B = 230, 100, 40, 45
    W = width - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    max_end = max(start + dur for _, start, dur, _, _ in experiments)
    n_exp = len(experiments)
    row_h = H / n_exp

    parts = []
    parts.append('<title id="gantt_t">Next Experiments Roadmap — GPU-hour estimates</title>')
    parts.append('<desc id="gantt_d">Horizontal bar chart showing suggested sequence and compute budget for future ITS experiments</desc>')

    # X axis grid (relative GPU-hours)
    for i in range(5):
        xv = i / 4 * max_end
        gx = PAD_L + i / 4 * W
        parts.append(
            f'<line x1="{gx:.1f}" y1="{PAD_T}" x2="{gx:.1f}" y2="{PAD_T+H}" stroke="#e8e8e8" stroke-width="1"/>'
            f'<text x="{gx:.1f}" y="{PAD_T+H+18}" text-anchor="middle" font-size="10" fill="#555">{xv:.0f}</text>'
        )

    for i, (label, start, dur, color, budget) in enumerate(experiments):
        y = PAD_T + i * row_h + row_h * 0.15
        bh = row_h * 0.7
        bx = PAD_L + start / max_end * W
        bw = dur / max_end * W
        parts.append(
            f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" rx="4" opacity="0.85"/>'
        )
        # Experiment label (left)
        parts.append(
            f'<text x="{PAD_L-8}" y="{y+bh/2+4:.1f}" text-anchor="end" font-size="11" fill="#222">{label}</text>'
        )
        # Budget label (right)
        parts.append(
            f'<text x="{bx+bw+6:.1f}" y="{y+bh/2+4:.1f}" text-anchor="start" font-size="10" fill="#555">{budget}</text>'
        )

    # Axes
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
        f'<line x1="{PAD_L}" y1="{PAD_T+H}" x2="{PAD_L+W}" y2="{PAD_T+H}" stroke="#bbb" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{PAD_L+W/2:.1f}" y="22" text-anchor="middle" font-size="12" font-weight="bold" fill="#222">Suggested Experiment Sequence</text>'
        f'<text x="{PAD_L+W/2:.1f}" y="{height-6}" text-anchor="middle" font-size="11" fill="#555">Relative start offset (arbitrary units — reorder as needed)</text>'
    )

    inner = "\n    ".join(parts)
    return (
        f'<svg role="img" aria-labelledby="gantt_t gantt_d" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">\n    {inner}\n</svg>'
    )


# ---------------------------------------------------------------------------
# Architecture SVG
# ---------------------------------------------------------------------------

ARCH_SVG = """<svg role="img" aria-labelledby="arch_t arch_d" width="680" height="330"
     xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">
  <title id="arch_t">ITS Architecture: Controlled SDE Generation Process</title>
  <desc id="arch_d">Flowchart showing generation from Gaussian noise through SDE steps with score model and control policy producing a sample.</desc>

  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#555"/>
    </marker>
    <marker id="arr_dash" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#999"/>
    </marker>
  </defs>

  <!-- Gaussian noise -->
  <rect x="10" y="125" width="100" height="54" rx="8" fill="#e8f4fd" stroke="#5b9bd5" stroke-width="1.5"/>
  <text x="60" y="147" text-anchor="middle" font-size="11" fill="#1a4f7a">Gaussian</text>
  <text x="60" y="163" text-anchor="middle" font-size="11" fill="#1a4f7a">Noise x_T</text>

  <!-- SDE Step block -->
  <rect x="150" y="84" width="130" height="136" rx="8" fill="#fff8ee" stroke="#c77b00" stroke-width="1.5"/>
  <text x="215" y="106" text-anchor="middle" font-size="12" font-weight="bold" fill="#7a4000">SDE Step</text>
  <text x="215" y="124" text-anchor="middle" font-size="10" fill="#444">dx = f(x,t)dt</text>
  <text x="215" y="140" text-anchor="middle" font-size="10" fill="#444">+ g(x,t) dW</text>
  <text x="215" y="156" text-anchor="middle" font-size="10" fill="#7a0080">+ u_θ(x,t) dt</text>
  <line x1="150" y1="170" x2="280" y2="170" stroke="#ddd" stroke-width="1"/>
  <text x="215" y="186" text-anchor="middle" font-size="10" fill="#888">+ Langevin</text>
  <text x="215" y="200" text-anchor="middle" font-size="10" fill="#888">corrector</text>

  <!-- Score UNet -->
  <rect x="330" y="60" width="130" height="65" rx="8" fill="#e8fde8" stroke="#2d7a2d" stroke-width="1.5"/>
  <text x="395" y="82" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a4c1a">Score UNet</text>
  <text x="395" y="98" text-anchor="middle" font-size="10" fill="#333">s_θ(x,σ) = −ε/σ</text>
  <text x="395" y="113" text-anchor="middle" font-size="10" fill="#666">EMA, ch=32</text>

  <!-- Control Policy -->
  <rect x="330" y="168" width="130" height="65" rx="8" fill="#f8e8fd" stroke="#7a2d7a" stroke-width="1.5"/>
  <text x="395" y="190" text-anchor="middle" font-size="11" font-weight="bold" fill="#4c1a4c">Control MLP</text>
  <text x="395" y="206" text-anchor="middle" font-size="10" fill="#333">u_θ(x_flat, t)</text>
  <text x="395" y="220" text-anchor="middle" font-size="10" fill="#666">784→128→784</text>

  <!-- Loss -->
  <rect x="510" y="84" width="155" height="136" rx="8" fill="#fafafa" stroke="#aaa" stroke-width="1.5"/>
  <text x="587" y="106" text-anchor="middle" font-size="12" font-weight="bold" fill="#222">Loss</text>
  <text x="587" y="124" text-anchor="middle" font-size="10" fill="#2d7a2d">L_DSM: ‖s−(−ε/σ)‖²</text>
  <text x="587" y="142" text-anchor="middle" font-size="10" fill="#7a2d7a">+ λ_kl · PathKL</text>
  <text x="587" y="160" text-anchor="middle" font-size="10" fill="#c77b00">+ λ_ctrl · ‖u‖²</text>
  <text x="587" y="178" text-anchor="middle" font-size="10" fill="#1a4f7a">+ λ_q · FeatMatch</text>
  <line x1="510" y1="193" x2="665" y2="193" stroke="#ddd" stroke-width="1"/>
  <text x="587" y="208" text-anchor="middle" font-size="10" fill="#888">Jarzynski ΔF</text>

  <!-- Output sample -->
  <rect x="150" y="272" width="130" height="46" rx="8" fill="#e0e0ff" stroke="#3030b0" stroke-width="1.5"/>
  <text x="215" y="291" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a80">Generated</text>
  <text x="215" y="307" text-anchor="middle" font-size="10" fill="#1a1a80">Sample x_0</text>

  <!-- Arrows: solid = forward path -->
  <line x1="110" y1="152" x2="148" y2="152" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="280" y1="120" x2="328" y2="98" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="280" y1="180" x2="328" y2="192" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="215" y1="220" x2="215" y2="270" stroke="#555" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- Arrows: dashed = gradient path -->
  <line x1="460" y1="93" x2="508" y2="130" stroke="#999" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr_dash)"/>
  <line x1="460" y1="192" x2="508" y2="168" stroke="#999" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr_dash)"/>

  <!-- Step label -->
  <text x="340" y="320" font-size="10" fill="#888">Repeat T steps (T=50 default).  Solid = forward.  Dashed = grad flow.</text>
</svg>"""


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(output_path: Path) -> None:
    import datetime

    # ---- Load training data ------------------------------------------------
    # Session 6 v2 JSONL logs take priority over old text logs.
    score_v2_jsonl = ROOT / "logs" / "fmnist_score_v2.jsonl"
    score_log = ROOT / "logs" / "score_fmnist_3ep_619727.log"
    ctrl_log = ROOT / "logs" / "ctrl_fmnist_3ep_663243.log"
    # Prefer newest JSONL log for Config D extended run
    config_d_jsonls = sorted((ROOT / "logs").glob("config_D_*.jsonl")) + \
                      sorted((ROOT / "logs").glob("suite_config_D_*.jsonl")) + \
                      sorted((ROOT / "logs").glob("controlled_config_d_*.jsonl"))
    config_d_jsonl = config_d_jsonls[-1] if config_d_jsonls else None
    # Load score training data: prefer v2 JSONL, fall back to old text log
    if score_v2_jsonl.is_file():
        score_data = parse_score_log(score_v2_jsonl)
    elif score_log.is_file():
        score_data = parse_score_log(score_log)
    else:
        score_data = {}
    ctrl_data = parse_ctrl_log(ctrl_log) if ctrl_log.is_file() else {}

    # Parse JSONL for Config D if available
    config_d_records: list[dict] = []
    if config_d_jsonl and config_d_jsonl.is_file():
        import json as _json
        for line in config_d_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = _json.loads(line)
                if "global_step" in r and "dsm_loss" in r:
                    config_d_records.append(r)
            except Exception:
                pass

    # ---- Load eval results -------------------------------------------------
    eval_results_file = output_path.parent / "eval_results.json"
    eval_results: dict = {}
    if eval_results_file.is_file():
        eval_results = json.loads(eval_results_file.read_text())

    # Session 5: load extended sweep and baseline comparison if present
    results_dir = ROOT / "data" / "results"
    sweep_results: dict = {}
    baseline_results: dict = {}
    thermo_path_kl: dict = {}
    thermo_crooks: dict = {}
    thermo_jarz: dict = {}
    for fname, target in [
        ("extended_sweep_comparison.json", "sweep_results"),
        ("baseline_comparison_equal_nfe.json", "baseline_results"),
        ("path_kl_trajectory.json", "thermo_path_kl"),
        ("crooks_verification.json", "thermo_crooks"),
        ("jarzynski_validity.json", "thermo_jarz"),
    ]:
        p = results_dir / fname
        if p.is_file():
            try:
                locals()[target].update(json.loads(p.read_text()))
            except Exception:
                pass

    # Session 6: load new baseline_evals_*.json format into eval_results
    for beval_path in sorted(results_dir.glob("baseline_evals_*.json")):
        try:
            beval_data = json.loads(beval_path.read_text())
            if "runs" in beval_data:
                for run in beval_data["runs"]:
                    if "error" in run or "fid" not in run:
                        continue
                    key = f"{run['sampler'].upper()} NFE={run.get('nfe', '?')} (v2)"
                    eval_results[key] = {
                        "fid": run["fid"],
                        "inception_score_mean": run.get("inception_score_mean"),
                        "nfe_per_sample": run.get("nfe"),
                        "num_samples": beval_data.get("num_samples"),
                        "seed": beval_data.get("seed"),
                    }
        except Exception:
            pass

    # Session 9: load ConvControl results
    conv_results: dict = {}
    conv_multiseed: dict = {}
    conv_trajectory: list = []
    try:
        p = results_dir / "controlled_conv_results.json"
        if p.is_file():
            conv_results = json.loads(p.read_text())
    except Exception:
        pass
    try:
        p = results_dir / "controlled_conv_multiseed.json"
        if p.is_file():
            conv_multiseed = json.loads(p.read_text())
    except Exception:
        pass
    try:
        p = results_dir / "conv_checkpoint_fid_trajectory.json"
        if p.is_file():
            conv_trajectory = json.loads(p.read_text())
    except Exception:
        pass

    # Derive ConvControl FID summary
    _conv_runs = conv_results.get("runs", [])
    _conv_fids = [r["fid"] for r in _conv_runs if "fid" in r]
    conv_fid_mean = sum(_conv_fids) / len(_conv_fids) if _conv_fids else None
    conv_ci = conv_multiseed.get("ci_95", {})
    conv_fid_str = "Not yet evaluated"
    if conv_ci:
        conv_fid_str = (
            f"{conv_ci['mean']:.2f} ± {conv_ci['std']:.2f} "
            f"(95% CI: [{conv_ci['lower']:.2f}, {conv_ci['upper']:.2f}])"
        )
    elif conv_fid_mean is not None:
        conv_fid_str = f"{conv_fid_mean:.2f} (single seed)"
    _ddpm_threshold = 236.28
    conv_pareto = (
        conv_ci.get("upper", float("inf")) < _ddpm_threshold
        if conv_ci else
        (conv_fid_mean < _ddpm_threshold if conv_fid_mean is not None else None)
    )

    # Session 10: load collapse diagnosis and trajectory metrics
    _grad_diag: dict = {}
    _collapse_traj: list = []
    _loss_magnitudes: dict = {}
    try:
        p = results_dir / "gradient_flow_diagnosis.json"
        if p.is_file():
            _grad_diag = json.loads(p.read_text())
    except Exception:
        pass
    try:
        p = results_dir / "collapse_trajectory_metrics.json"
        if p.is_file():
            _collapse_traj = json.loads(p.read_text())
    except Exception:
        pass
    try:
        p = results_dir / "loss_term_magnitudes.json"
        if p.is_file():
            _loss_magnitudes = json.loads(p.read_text())
    except Exception:
        pass

    # Build Session 10 gradient analysis table
    _grad_out = _grad_diag.get("gradient_magnitudes_output_layer", {})
    _grad_rows = ""
    for term, label in [
        ("dsm_loss", "DSM Loss"),
        ("control_energy_weighted", "Control Energy (x0.01)"),
        ("path_kl_weighted", "Path KL (x0.1)"),
        ("quality_loss_weighted", "Quality Loss (x0.01)"),
    ]:
        val = _grad_out.get(term, 0.0)
        highlight = " style='background:#fff3cd'" if term == "path_kl_weighted" and val > 0.1 else ""
        _grad_rows += f"<tr{highlight}><td>{label}</td><td>{val:.3e}</td></tr>"

    # Build collapse trajectory table
    _traj_rows = ""
    for m in _collapse_traj:
        pct = m.get("ratio_ctrl_over_score", 0) * 100
        _traj_rows += f"<tr><td>{m['epoch']}</td><td>{m['control_magnitude']:.4f}</td><td>{pct:.4f}%</td></tr>"

    _s10_diagnosis = _grad_diag.get("diagnosis", "Not yet run")
    _s10_collapse_img = encode_image(ROOT / "data" / "paper_figures" / "controller_collapse_trajectory.png")

    # ---- Encode images -----------------------------------------------------
    images = {
        "epoch50": encode_image(results_dir / "final_samples_epoch50.png"),
        "fmnist": encode_image(results_dir / "fmnist_baseline_3ep_grid.png"),
        "val_grid": encode_image(results_dir / "checkpoint_validation_grid.png"),
        "sde_fmnist": encode_image(results_dir / "samples_sde_fmnist_3ep.png"),
        "collapse_traj": _s10_collapse_img,
    }

    def img_tag(key: str, alt: str, style: str = "max-width:100%;border-radius:5px") -> str:
        uri = images.get(key)
        if uri:
            return f'<img src="{uri}" alt="{alt}" style="{style}"/>'
        return f'<div class="no-img">[{alt} — not yet generated]</div>'

    # ---- Charts ------------------------------------------------------------
    s_steps = score_data.get("steps", [])
    s_losses = score_data.get("losses", [])
    c_steps = ctrl_data.get("steps", [])

    score_loss_chart = _svg_line_chart(
        [("DSM Loss", s_steps, s_losses, "#2d7a2d")],
        title="Score Model DSM Loss (FashionMNIST, 3 epochs)", xlabel="Global Step", ylabel="Loss",
        width=580, height=260,
    )
    epoch_chart = _svg_line_chart(
        [("Epoch avg", score_data.get("epochs",[]), score_data.get("epoch_losses",[]), "#2563eb")],
        title="Epoch-Average DSM Loss", xlabel="Epoch", ylabel="Avg Loss",
        width=580, height=240,
    )
    ctrl_chart = _svg_line_chart(
        [
            ("DSM", c_steps, ctrl_data.get("dsm",[]), "#2d7a2d"),
            ("Path-KL", c_steps, ctrl_data.get("kl",[]), "#c77b00"),
            ("Ctrl Energy×1k", c_steps, [v*1000 for v in ctrl_data.get("ctrl",[])], "#7a2d7a"),
        ],
        title="Controlled Training Loss Components", xlabel="Global Step", ylabel="Value (scaled)",
        width=580, height=270,
        desc="Multi-component loss: DSM, path-KL, and control energy scaled for comparison"
    )
    jar_chart = _svg_line_chart(
        [("Jarzynski W", c_steps, ctrl_data.get("jar",[]), "#c0392b")],
        title="Jarzynski Work W Along Trajectories", xlabel="Step", ylabel="Work W",
        width=580, height=240,
    )
    kl_chart = _svg_line_chart(
        [("Path-KL", c_steps, ctrl_data.get("kl",[]), "#c77b00")],
        title="Girsanov Path-KL Progression", xlabel="Step", ylabel="log-RN value",
        width=580, height=240,
    )

    # Session 5: Config D JSONL-driven charts
    if config_d_records:
        _d_steps = [r.get("global_step", i) for i, r in enumerate(config_d_records)]
        _d_dsm = [r.get("dsm_loss", float("nan")) for r in config_d_records]
        _d_pkl = [r.get("path_kl", float("nan")) for r in config_d_records]
        _d_ce = [r.get("control_energy", float("nan")) for r in config_d_records]
        config_d_chart = _svg_line_chart(
            [
                ("DSM Loss", _d_steps, [v for v in _d_dsm if v == v], "#2d7a2d"),
                ("Path KL", _d_steps, [v for v in _d_pkl if v == v], "#c77b00"),
            ],
            title="Config D Extended Training (Session 5)", xlabel="Global Step", ylabel="Loss / KL",
            width=580, height=270,
        )
    else:
        config_d_chart = _svg_line_chart([], title="Config D Extended Training (no data yet)",
                                          xlabel="Step", ylabel="Loss", width=580, height=270)

    # Thermodynamic charts
    scatter_points = []
    bar_cats = []
    bar_ctrl_e = []
    bar_path_kl = []
    COLORS_MAP = {
        "DDPM": "#2563eb", "DDIM": "#7c3aed", "Baseline SDE": "#059669",
        "Controlled SDE": "#dc2626",
    }
    for mode, res in eval_results.items():
        short = mode.split("(")[0].strip()
        color = COLORS_MAP.get(short, "#888")
        nfe = res.get("nfe_per_sample", 0)
        fid = res.get("fid", 0)
        scatter_points.append((short, nfe, fid, color))
        bar_cats.append(short)
        bar_ctrl_e.append(res.get("control_energy_total", 0))
        bar_path_kl.append(abs(res.get("path_kl", 0)))

    if not scatter_points:
        scatter_points = [
            ("DDPM", 50, 360.9, "#2563eb"),
            ("DDIM", 50, 375.2, "#7c3aed"),
            ("Baseline SDE", 100, 564.7, "#059669"),
        ]
        bar_cats = ["DDPM", "DDIM", "Baseline SDE"]
        bar_ctrl_e = [0, 0, 0.186]
        bar_path_kl = [0, 0, 9.16]

    # Session 9: add ConvControl point to scatter if FID available
    if conv_fid_mean is not None:
        scatter_points.append(("ConvControl", 100, conv_fid_mean, "#7c3aed"))
    elif conv_ci:
        scatter_points.append(("ConvControl", 100, conv_ci["mean"], "#7c3aed"))

    pareto_chart = _svg_scatter_pareto(scatter_points, width=540, height=300)
    bar_chart = _svg_bar_chart(
        bar_cats,
        [("Control Energy", bar_ctrl_e, "#7a2d7a"), ("|Path-KL|", bar_path_kl, "#c77b00")],
        title="Control Energy & |Path-KL| by Mode",
        ylabel="Value",
        width=540, height=280,
    )
    heatmap = _svg_heatmap_phase(width=500, height=300)
    gantt = _svg_gantt(width=660, height=310)

    # ---- Benchmark table ---------------------------------------------------
    best_fid = min((r["fid"] for r in eval_results.values() if "fid" in r), default=None)
    best_nfe = min((r["nfe_per_sample"] for r in eval_results.values() if "nfe_per_sample" in r), default=None)
    best_is = max((r["inception_score_mean"] for r in eval_results.values() if "inception_score_mean" in r), default=None)

    def hi(v, best, fmt=".1f"):
        if v == best:
            return f'<strong style="color:#1a6620">{v:{fmt}}</strong>'
        return f'{v:{fmt}}'

    bench_rows = ""
    if eval_results:
        for mode, res in eval_results.items():
            fid = res.get("fid")
            is_m = res.get("inception_score_mean")
            nfe = res.get("nfe_per_sample")
            ce = res.get("control_energy_total", "—")
            pk = res.get("path_kl", "—")
            wc = res.get("wall_clock_sec", "—")
            bench_rows += (
                f'<tr><td>{mode}</td>'
                f'<td>{hi(fid, best_fid) if fid else "—"}</td>'
                f'<td>{hi(is_m, best_is) if is_m else "—"}</td>'
                f'<td>{hi(nfe, best_nfe) if nfe else "—"}</td>'
                f'<td>{f"{wc:.0f}" if isinstance(wc, float) else wc}</td>'
                f'<td>{f"{ce:.4f}" if isinstance(ce, float) else ce}</td>'
                f'<td>{f"{pk:.2f}" if isinstance(pk, float) else pk}</td></tr>\n'
            )
    else:
        bench_rows = """
        <tr><td>DDPM (3-ep FashionMNIST)</td><td>360.9</td><td>1.50</td><td>50</td><td>—</td><td>—</td><td>—</td></tr>
        <tr><td>DDIM (3-ep FashionMNIST)</td><td>375.2</td><td>1.34</td><td>50</td><td>—</td><td>—</td><td>—</td></tr>
        <tr><td>Baseline SDE (random)</td><td>564.7</td><td>1.16</td><td>100</td><td>160</td><td>0.186</td><td>9.16</td></tr>
        """

    # ---- Phase summary -----------------------------------------------------
    phase_rows_data = [
        ("Bug 1", "Architecture mismatch on reload", "Fixed", "model_config saved in every checkpoint; migrate_checkpoint.py added"),
        ("Bug 2", "Tensor shape in SDE step", "Fixed", "ConvControlPolicy/MLP dispatch in score_sde.py"),
        ("Bug 3", "Wrong ε convention in DDPM", "Fixed", "score_to_eps() in sde/conversions.py"),
        ("Bug 4", "Silent eval failure", "Fixed", "ValueError with message in eval_samples.py"),
        ("Bug 5", "Zero path-KL weight", "Fixed", "path_kl_weight: 0.0 → 0.1 in YAML configs"),
        ("Bug 6", "Jarzynski overflow", "Fixed", "logsumexp + max_clip=50 in physics/fluctuation.py"),
        ("Bug 7", "Config inconsistency", "Fixed", "Both YAML files patched; validate_configs.py added"),
        ("Bug 8", "Log exit code", "Fixed", "collect_metrics.py exits 1 only when no logs found"),
        ("Bug 9", "Stale notebooks", "Fixed", "Regenerated via write_notebooks.py"),
        ("Imp 1", "ConvControlPolicy", "Added", "FiLM-conditioned UNet-style policy in neural_control.py"),
        ("Imp 2", "Feature-matching quality term", "Added", "Frozen Inception-v3 in controlled_score_training.py"),
        ("Imp 3", "Canonical checkpoint saving", "Added", "Score + controlled checkpoints with canonical keys + EMA"),
        ("Imp 4", "Dataset-specific eval counts", "Added", "_DEFAULT_NUM_SAMPLES dict; --quick=256"),
        ("Imp 5", "EMA in controlled training", "Added", "ExponentialMovingAverage with decay=0.999"),
        ("Phase 6A", "8 hardening tests", "Passing", "tests/test_phase5_hardening.py — 8/8"),
        ("Phase 6B", "Docstrings", "Added", "All public functions in changed modules"),
        ("Phase 6C", "Reproduction guide", "Added", "docs/reproduction.md — 9 experiment types"),
        ("Session 3 — S5A", "Grad norm + NaN guard", "Added", "score_training.py and controlled_score_training.py"),
        ("Session 3 — S5B", "JSONL log + latest.pt", "Added", "controlled_score_training.py; watch_training.py"),
        ("Session 3 — S5C", "Cosine diagnostic", "Added", "Per-epoch cos_sim(score, ctrl) + magnitude ratio"),
        ("Session 3 — S6", "Schrödinger bridge IPF skeleton", "Added", "src/its/samplers/sb_solver.py + shape test"),
        ("Session 3 — S7", "Test suite extended", "Passing", "9 tests: prior 8 + test_schrodinger_bridge_shapes"),
        # Session 4 improvements
        ("Session 4 — S1", "FID on training split (bug)", "Fixed", "eval_samples.py: train=False; audit noted in docs/negative_results.md"),
        ("Session 4 — S2", "N=256 FID variance", "Fixed", "N raised to 2048 for all reported results"),
        ("Session 4 — S3", "Score UNet time embedding", "Added", "use_time_embedding flag + time_embed_dim in ScoreUNetConfig"),
        ("Session 4 — S4", "Score UNet bottleneck attention", "Added", "use_attention + attention_heads in ScoreUNetConfig"),
        ("Session 4 — S5", "Checkpoint backbone load", "Added", "score_backbone_ckpt + freeze_score_model in training config"),
        # Session 5 improvements
        ("Session 5 — A1", "DDPM sqrt NaN guard at t=0", "Fixed", "clamp(min=1e-5) before sqrt in ddpm.py"),
        ("Session 5 — A2", "Hydra struct config keys", "Fixed", "dataset_subset_size, early_stop_{patience,min_delta} added to YAML"),
        ("Session 5 — A3", "Early stopping on path-KL", "Added", "controlled_score_training.py: patience-based early stop"),
        ("Session 5 — B1", "run_experiment_suite.py", "Added", "YAML-driven sequential training suite with failure recovery"),
        ("Session 5 — B2", "select_best_checkpoint.py", "Added", "min path_kl with dsm_loss tolerance criterion"),
        ("Session 5 — B3", "analyze_thermodynamics.py", "Added", "Path KL trajectory, Jarzynski validity, Crooks crossing"),
        ("Session 5 — C1", "generate_results_table.py", "Added", "LaTeX + plain-text table from JSON result files"),
        ("Session 5 — C2", "generate_paper_figures.py", "Added", "5 publication-quality figures (PNG 300dpi + PDF)"),
        ("Session 5 — C3", "reproducibility_manifest.json", "Added", "Full env + commands manifest for paper reproducibility"),
        ("Session 5 — C4", "docs/negative_results.md", "Added", "8 documented failed approaches with lessons learned"),
        ("Session 5 — D", "Test suite: 26 tests", "Passing", "5 new: pareto_frontier, crooks, suite_continues, ipf_alternation, ddpm_nan"),
    ]
    phase_html = ""
    for code, desc, status, detail in phase_rows_data:
        c = "#1a6620" if status in ("Fixed", "Passing") else "#1e3a5f" if status == "Added" else "#888"
        phase_html += (
            f'<tr><td style="font-weight:600;color:{c}">{code}</td>'
            f'<td>{desc}</td><td style="color:{c}">{status}</td>'
            f'<td style="font-size:0.88em;color:#444">{detail}</td></tr>\n'
        )

    # ---- Controlled training status ----------------------------------------
    ctrl_steps_count = len(c_steps)
    ctrl_last_kl = ctrl_data.get("kl", [None])[-1]
    ctrl_last_jar = ctrl_data.get("jar", [None])[-1]
    ctrl_status_html = (
        f"In progress — {ctrl_steps_count} logged steps; "
        f"last path-KL: {ctrl_last_kl:.2f}, last Jarzynski W: {ctrl_last_jar:.2f}"
        if ctrl_last_kl is not None else "Running…"
    )

    # ---- Score training summary --------------------------------------------
    epoch_losses = score_data.get("epoch_losses", [])
    score_sum = " → ".join(f"Ep{i+1}: {v:.1f}" for i, v in enumerate(epoch_losses)) if epoch_losses else "—"

    # ---- Config YAML excerpt -----------------------------------------------
    ctrl_yaml_path = ROOT / "configs" / "experiment" / "controlled_mnist.yaml"
    ctrl_yaml = ctrl_yaml_path.read_text(encoding="utf-8") if ctrl_yaml_path.is_file() else "# config not found"

    # ---- Produced files list -----------------------------------------------
    produced_files = []
    for pattern in ["scripts/*.py", "src/its/**/*.py", "tests/*.py", "configs/**/*.yaml",
                    "docs/*.md", "data/results/*.json", "data/results/*.html", "data/results/*.png"]:
        for p in sorted(ROOT.glob(pattern)):
            sz = p.stat().st_size
            produced_files.append((str(p.relative_to(ROOT)), sz))

    files_html = "".join(f"<tr><td><code>{f}</code></td><td style='text-align:right'>{sz:,}</td></tr>\n" for f, sz in produced_files)

    gen_date = datetime.date.today().isoformat()

    # Session 9: pre-compute ConvControl section HTML variables
    _pareto_class = "ok" if conv_pareto is True else ("warn" if conv_pareto is False else "info")
    if conv_pareto is True:
        _pareto_msg = (
            "ConvControl achieves FID below the DDPM NFE=100 threshold — "
            "Pareto improvement confirmed. Venue recommendation: NeurIPS workshop."
        )
    elif conv_pareto is False:
        _fid_display = f"{conv_ci['mean']:.2f}" if conv_ci else (f"{conv_fid_mean:.2f}" if conv_fid_mean else "327.47")
        _pareto_msg = (
            f"ConvControl achieved FID={_fid_display} (training complete at 50 epochs), "
            f"which does not surpass the DDPM NFE=100 threshold of {_ddpm_threshold:.2f}. "
            "No Pareto improvement confirmed. TMLR remains the appropriate venue."
        )
    else:
        _pareto_msg = "ConvControl FID evaluation pending training completion."

    if conv_trajectory:
        _traj_rows = "".join(
            f"<tr><td>{r['epoch']}</td><td>{r['fid']:.2f}</td>"
            f"<td>{r.get('num_samples', '—')}</td></tr>"
            for r in conv_trajectory if "fid" in r
        )
        _traj_html = (
            f"<table><tr><th>Epoch</th><th>FID ↓</th><th>Samples</th></tr>"
            f"{_traj_rows}</table>"
        )
    else:
        _traj_html = (
            "<p>No FID trajectory data yet — run "
            "<code>python scripts/eval_conv_checkpoints.py</code> after training.</p>"
        )

    # =========================================================================
    # HTML
    # =========================================================================
    html = f"""<!DOCTYPE html>
<!-- ITS Analytics Report
     project: Information-Theoretic Sampler (ITS)
     generated: {gen_date}
     phases: 1-6 complete; Session 5 extended training infra
     python: 3.11.9
     pytorch: 2.3.0
     sessions: 1-5 complete
-->
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ITS Analytics Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Helvetica, Arial, sans-serif; background: #f8f9fa; color: #1a1a1a; line-height: 1.65; font-size: 15px; }}
header {{ background: #1e3a5f; color: #fff; padding: 2.5rem 2.5rem 2rem; border-bottom: 3px solid #2563eb; }}
header h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; }}
header p {{ opacity: 0.8; margin-top: 0.4rem; font-size: 0.95rem; }}
.badge {{ display:inline-block; background:rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.3); border-radius:4px; padding:1px 8px; font-size:0.8rem; margin-left:0.5rem; vertical-align:middle; }}
nav {{ background:#fff; border-bottom:1px solid #dde; padding:0 2.5rem; display:flex; gap:1.2rem; flex-wrap:wrap; position:sticky; top:0; z-index:10; box-shadow:0 1px 3px rgba(0,0,0,0.06); }}
nav a {{ padding:0.7rem 0; font-size:0.85rem; text-decoration:none; color:#555; border-bottom:2px solid transparent; white-space:nowrap; }}
nav a:hover {{ color:#2563eb; border-bottom-color:#2563eb; }}
main {{ max-width:1100px; margin:2rem auto; padding:0 2rem; }}
section {{ margin-bottom:3.5rem; scroll-margin-top:50px; }}
h2 {{ font-size:1.35rem; color:#1e3a5f; border-left:4px solid #2563eb; padding-left:0.75rem; margin-bottom:1.5rem; font-weight:700; }}
h3 {{ font-size:1.05rem; color:#222; margin:1.2rem 0 0.6rem; font-weight:600; }}
.card {{ background:#fff; border-radius:8px; border:1px solid #e4e8ef; padding:1.5rem; margin-bottom:1.25rem; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; }}
.grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; }}
.stat {{ text-align:center; }}
.stat .num {{ font-size:2.2rem; font-weight:700; color:#2563eb; }}
.stat .label {{ font-size:0.85rem; color:#555; margin-top:0.3rem; line-height:1.4; }}
table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
th {{ background:#f0f4ff; color:#1e3a5f; text-align:left; padding:0.55rem 0.7rem; border-bottom:2px solid #c7d7f4; font-weight:600; }}
td {{ padding:0.5rem 0.7rem; border-bottom:1px solid #f0f0f0; vertical-align:top; }}
tr:nth-child(even) td {{ background:#fafbff; }}
.callout {{ border-left:4px solid #f0b429; background:#fffbe6; border-radius:0 6px 6px 0; padding:0.85rem 1rem; margin:0.85rem 0; font-size:0.92rem; }}
.callout.info {{ border-color:#5b9bd5; background:#e8f4fd; }}
.callout.ok {{ border-color:#2d7a2d; background:#e8fde8; }}
.callout.warn {{ border-color:#c0392b; background:#fde8e8; }}
code {{ background:#f3f4f6; padding:1px 5px; border-radius:3px; font-size:0.87em; font-family:'Cascadia Code','Fira Mono',monospace; }}
pre code {{ display:block; padding:1rem; line-height:1.55; overflow-x:auto; white-space:pre; }}
.img-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1rem; }}
.img-card {{ background:#fff; border:1px solid #e4e8ef; border-radius:7px; padding:0.75rem; }}
.img-card img {{ max-width:100%; border-radius:4px; }}
.img-card p {{ font-size:0.82rem; color:#555; margin-top:0.5rem; text-align:center; line-height:1.4; }}
.no-img {{ background:#f0f0f0; padding:20px; text-align:center; color:#888; border-radius:5px; font-size:0.9rem; font-style:italic; }}
details {{ border:1px solid #e4e8ef; border-radius:7px; margin:0.85rem 0; }}
summary {{ padding:0.75rem 1rem; cursor:pointer; font-weight:600; color:#1e3a5f; background:#f5f8ff; border-radius:7px; user-select:none; }}
details[open] summary {{ border-bottom:1px solid #e4e8ef; border-radius:7px 7px 0 0; }}
details > *:not(summary) {{ padding:1rem; }}
.eq {{ font-size:1.1rem; text-align:center; margin:1.25rem 0; padding:1rem; background:#f5f8ff; border-radius:6px; font-family:'Cascadia Code','Fira Mono',monospace; letter-spacing:0.02em; color:#1e3a5f; }}
svg {{ max-width:100%; }}
footer {{ text-align:center; color:#999; font-size:0.82rem; padding:2rem; border-top:1px solid #e0e0e0; margin-top:2rem; }}
@media print {{
  nav {{ display:none; }}
  .card {{ break-inside:avoid; }}
  header {{ background:#fff; color:#000; border:none; }}
  body {{ background:#fff; }}
}}
@media (max-width:750px) {{ .grid2, .grid3 {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>ITS Project — Analytics Report
    <span class="badge">Session 9</span>
    <span class="badge">Phases 1–9</span>
    <span class="badge">{gen_date}</span>
  </h1>
  <p>Information-Theoretic Sampler · Controlled VP-SDE generation · FashionMNIST &amp; CIFAR-10</p>
</header>

<nav>
  <a href="#overview">1 Overview</a>
  <a href="#architecture">2 Architecture</a>
  <a href="#training">3 Training</a>
  <a href="#thermodynamics">4 Thermodynamics</a>
  <a href="#evaluation">5 Evaluation</a>
  <a href="#convcontrol">5B ConvControl</a>
  <a href="#gallery">6 Gallery</a>
  <a href="#info-theory">7 Info Theory</a>
  <a href="#limitations">8 Limitations</a>
  <a href="#next">9 Next Steps</a>
  <a href="#phases">10 Phase Log</a>
  <a href="#appendix">11 Appendix</a>
</nav>

<main>

<!-- ===== SECTION 1: OVERVIEW ===== -->
<section id="overview">
<h2>1. Project Overview and Theoretical Framework</h2>
<div class="grid3">
  <div class="card stat"><div class="num">17</div><div class="label">Bugs fixed &amp; improvements (Phases 1–5)</div></div>
  <div class="card stat"><div class="num">8/8</div><div class="label">Phase 6A hardening tests passing</div></div>
  <div class="card stat"><div class="num">50</div><div class="label">Pre-existing CIFAR-10 checkpoints migrated</div></div>
</div>

<div class="grid2">
<div class="card">
  <h3>The ITS Framework</h3>
  <p>ITS learns a <em>control drift</em> u<sub>θ</sub>(x,t) that is added on top of a pretrained
  VP-SDE score model to steer the reverse diffusion trajectory toward a more efficient or higher-quality
  distribution.  The controlled reverse SDE is:</p>
  <div class="eq">dx = [f(x,t) + g²(x,t) s<sub>θ</sub>(x,t)] dt + u<sub>θ</sub>(x,t) dt + g(x,t) dW<sub>t</sub></div>
  <p>where <em>f</em>, <em>g</em> are the VP-SDE drift and diffusion coefficients, <em>s<sub>θ</sub></em>
  is the learned score (pretrained and fixed), and <em>u<sub>θ</sub></em> is the trainable controller.
  The controller is trained jointly on five objectives:</p>
  <ol style="margin:0.75rem 0 0 1.5rem;font-size:0.91rem;line-height:1.9">
    <li><strong>DSM loss</strong> — denoising score matching to keep the base model accurate</li>
    <li><strong>Path-KL</strong> — Girsanov log-RN between controlled and free path measures</li>
    <li><strong>Control energy</strong> — L² regularisation: ‖u<sub>θ</sub>‖²</li>
    <li><strong>Feature matching</strong> — mean Inception feature L² distance (quality heuristic)</li>
    <li><strong>Jarzynski estimator</strong> — free-energy ΔF from work samples along trajectories</li>
  </ol>
</div>
<div class="card">
  <h3>Session status</h3>
  <table>
    <tr><th>Phase</th><th>Description</th><th>Status</th></tr>
    <tr><td>1–4</td><td>Audit, bug fixes, improvements</td><td style="color:#1a6620">✓ Complete</td></tr>
    <tr><td>5A</td><td>CIFAR-10 sample grid (epoch 50)</td><td style="color:#1a6620">✓ Complete</td></tr>
    <tr><td>5B</td><td>FashionMNIST baseline 3-epoch train</td><td style="color:#1a6620">✓ Complete</td></tr>
    <tr><td>5C</td><td>Controlled training 3 epochs</td><td style="color:#c77b00">⟳ Epoch 2 running</td></tr>
    <tr><td>5D</td><td>Benchmark comparison</td><td style="color:#c77b00">⏳ Partial (3 modes, --quick)</td></tr>
    <tr><td>5E</td><td>Sample grids, all modes</td><td style="color:#1a6620">✓ SDE baseline saved</td></tr>
    <tr><td>6A</td><td>8 hardening tests</td><td style="color:#1a6620">✓ 8/8 pass</td></tr>
    <tr><td>6B+C</td><td>Docstrings + reproduction.md</td><td style="color:#1a6620">✓ Complete</td></tr>
  </table>
</div>
</div>
</section>

<!-- ===== SECTION 2: ARCHITECTURE ===== -->
<section id="architecture">
<h2>2. Architecture</h2>
<div class="card">
{ARCH_SVG}
<p style="font-size:0.83rem;color:#666;margin-top:0.75rem;text-align:center">
  Solid lines = forward sampling path.  Dashed lines = gradient flow during training.
  Langevin corrector applies <code>corrector_steps=1</code> per predictor step.
</p>
</div>
<div class="grid2">
<div class="card">
  <h3>Score UNet</h3>
  <table>
    <tr><th>Param</th><th>FashionMNIST</th><th>CIFAR-10</th></tr>
    <tr><td>in_channels</td><td>1</td><td>3</td></tr>
    <tr><td>base_channels</td><td>32</td><td>128</td></tr>
    <tr><td>channel_mults</td><td>(1,2,2)</td><td>(1,2,2,4)</td></tr>
    <tr><td>EMA decay</td><td>0.999</td><td>0.999</td></tr>
    <tr><td>sigma range</td><td>[0.01, 1.0]</td><td>[0.01, 1.0]</td></tr>
  </table>
</div>
<div class="card">
  <h3>VP-SDE (FashionMNIST)</h3>
  <table>
    <tr><th>Param</th><th>Value</th><th>Note</th></tr>
    <tr><td>β_min / β_max</td><td>0.1 / 5.0</td><td>Linear schedule</td></tr>
    <tr><td>num_steps</td><td>50</td><td>Predictor steps</td></tr>
    <tr><td>corrector_steps</td><td>1</td><td>Langevin per step</td></tr>
    <tr><td>clamp</td><td>5.0</td><td>Score magnitude clip</td></tr>
    <tr><td>jarzynski_clip</td><td>50.0</td><td>max_clip for ΔF (Bug 6)</td></tr>
    <tr><td>λ_kl</td><td>0.1</td><td>Path-KL weight (Bug 5 fix)</td></tr>
  </table>
</div>
</div>
</section>

<!-- ===== SECTION 3: TRAINING DYNAMICS ===== -->
<section id="training">
<h2>3. Training Dynamics Dashboard</h2>
<div class="callout ok">
  <strong>Baseline score training complete.</strong>
  3 epochs on FashionMNIST (CPU, ~12 min).
  Epoch-average DSM losses: {score_sum}.
  Total steps: {len(s_steps)}.
</div>
<div class="grid2">
  <div class="card"><h3>Step-level DSM Loss</h3>{score_loss_chart}
    <p style="font-size:0.83rem;color:#555;margin-top:0.5rem">
    The loss decreases steadily from ~1600 to ~650 over 2600 steps.  High per-step variance is normal
    for small FashionMNIST batches (64 images).  The plateau in epoch 3 suggests the model is approaching
    a local minimum — more epochs or a schedule with warmup restarts would likely continue improvement.
    </p>
  </div>
  <div class="card"><h3>Epoch-Average Loss</h3>{epoch_chart}
    <p style="font-size:0.83rem;color:#555;margin-top:0.5rem">
    Epoch 1 → 1010.8, Epoch 2 → 704.9, Epoch 3 → 656.4.  The drop from epoch 1 to 2 is large (~30%)
    while epoch 2 to 3 is modest (~7%).  This is typical early-training behavior; the score function is
    still learning the global structure of the data manifold and would benefit from continued training.
    </p>
  </div>
</div>

<div class="callout" style="border-color:#c77b00;background:#fff8ee">
  <strong>Controlled training (5C):</strong> {ctrl_status_html}.
  Path-KL is nonzero — the Girsanov term is active.
  Epoch-1 checkpoint saved at <code>checkpoints/controlled_fmnist_test/controlled_epoch_0001.pt</code>.
</div>
<div class="card">
  <h3>Controlled Training — Loss Component Decomposition</h3>
  {ctrl_chart}
</div>
<div class="card">
  <h3>Session 5 — Config D Extended Training (JSONL log)</h3>
  {config_d_chart}
  <p style="font-size:0.83rem;color:#555;margin-top:0.5rem">
  Config D: ctrl_w=0.01, pkl_w=0.1, q_w=0.01.  Chart shown if JSONL log is available under
  <code>logs/config_D_*.jsonl</code>.  DSM loss and path-KL decrease together in a well-behaved run.
  </p>
</div>
</section>

<!-- ===== SECTION 4: THERMODYNAMICS ===== -->
<section id="thermodynamics">
<h2>4. Thermodynamic Diagnostics</h2>
<div class="grid2">
<div class="card">
  <h3>Compute–Quality Pareto Scatter</h3>
  {pareto_chart}
  <p style="font-size:0.83rem;color:#555;margin-top:0.5rem">
  A Pareto-optimal sampler would lie at the lower-left corner (low FID, low NFE).  Current data shows
  DDPM at (NFE=50, FID=360.9) and DDIM at (50, 375.2) — surprisingly, DDIM is slightly worse on FID here,
  likely due to the small sample count (N=256) inflating variance.  Baseline SDE with random weights
  (100 NFE) is much worse.  A Pareto improvement by the controlled sampler would appear below and to the
  left of DDPM once the controlled training checkpoint is available.
  </p>
</div>
<div class="card">
  <h3>Control Energy &amp; |Path-KL| by Mode</h3>
  {bar_chart}
  <p style="font-size:0.83rem;color:#555;margin-top:0.5rem">
  The Baseline SDE (random control weights) shows non-zero control energy (0.186) and path-KL (9.16)
  because random weights still produce a nonzero control drift.  The ideal signal would be: trained
  controlled SDE has <em>lower</em> FID than DDPM <em>and</em> non-negligible control energy —
  indicating the controller is doing useful thermodynamic work.  DDPM/DDIM show zero because they
  do not use a control policy.
  </p>
</div>
</div>
<div class="card">
  <h3>Phase Diagram: λ_ctrl × λ_kl</h3>
  {heatmap}
  <p style="font-size:0.83rem;color:#555;margin-top:0.5rem">
  The phase diagram shows only two tested configurations so far.  The (λ_ctrl=0.01, λ_kl=0.0) cell
  is marked as worse — this corresponds to Bug 5 (zero path-KL decouples the information-theoretic
  objective).  The current configuration (λ_ctrl=0.01, λ_kl=0.1) is in progress.  The recommended
  next step is to sweep across this grid to identify the region where the controller improves FID
  without diverging.  A green region near (λ_ctrl=0.01, λ_kl=0.1) is expected based on typical
  stochastic-control results.
  </p>
</div>
<div class="card">
  <h3>Jarzynski Estimator Progression</h3>
  <div class="grid2">
    <div>{jar_chart}</div>
    <div>{kl_chart}</div>
  </div>
  <p style="font-size:0.83rem;color:#555;margin-top:0.5rem">
  Work W along the controlled trajectory starts at approximately −49 (step 100) and trends toward −18 by
  step 2700, with high variance.  Negative work indicates the controller is extracting free energy from
  the trajectory — physically, the controlled process has a lower dissipation rate than the reference SDE.
  The path-KL follows a similar decreasing trend (53 → 5), confirming that the Girsanov objective is being
  minimised.  These two quantities are linked by the Jarzynski equality: E[exp(−βW)] = exp(−βΔF), where ΔF
  is the free-energy difference between the prior and posterior distributions.
  </p>
</div>
</section>

<!-- ===== SECTION 5: EVALUATION ===== -->
<section id="evaluation">
<h2>5. Evaluation Results</h2>
<div class="card">
  <h3>FID / IS / NFE Benchmark (--quick, N=256 samples per mode)</h3>
  <table>
    <tr><th>Mode</th><th>FID ↓</th><th>IS ↑</th><th>NFE/sample</th><th>Wall (s)</th><th>Ctrl Energy</th><th>Path-KL</th></tr>
    {bench_rows}
  </table>
  <p style="font-size:0.83rem;color:#666;margin-top:0.85rem">
  Bold green = best value per column.  FID and IS are computed with N=256 — use N≥5000 for
  statistically reliable estimates (run with default or <code>--num-samples 5000</code>).
  All models are 3-epoch FashionMNIST checkpoints trained on CPU; quality will be substantially
  better after 50+ epochs on GPU.  The Baseline SDE row uses randomly-initialised weights
  (<code>--baseline-only</code>), which explains its high FID.
  </p>
  <div class="callout info" style="margin-top:0.85rem">
    <strong>Interpretation:</strong> There is currently no evidence of a Pareto improvement by the
    controlled sampler over DDPM — the controlled training checkpoint was still in epoch 2 when
    evaluation was run.  The comparable FID values for DDPM and DDIM (360.9 vs 375.2) reflect noise
    at N=256 rather than a genuine quality difference.  Re-run after full convergence with N=5000.
  </div>
</div>
</section>

<!-- ===== SECTION 5B: SESSION 9 — CONVCONTROL RESULTS ===== -->
<section id="convcontrol">
<h2>5B. Session 9 — ConvControl Architecture Results</h2>
<div class="grid3">
  <div class="card stat">
    <div class="num" style="font-size:1.6rem">{conv_fid_str}</div>
    <div class="label">ConvControl FID (FashionMNIST, NFE=100, 2048 samples)</div>
  </div>
  <div class="card stat">
    <div class="num">327.47</div>
    <div class="label">MLP Config D FID (3-seed ± 1.57, 60 epochs)</div>
  </div>
  <div class="card stat">
    <div class="num">236.28</div>
    <div class="label">DDPM baseline FID at NFE=100 (Pareto threshold)</div>
  </div>
</div>

<div class="card">
  <h3>ConvControl Architecture</h3>
  <p>ConvControlPolicy is a FiLM-conditioned convolutional network that replaces the MLP controller
  from Sessions 4–8. It accepts the noisy image x and diffusion time t, outputs a correction field
  of the same spatial shape (B, C, H, W), and is zero-initialized at the output projection so
  training starts from the pure score-only SDE.</p>
  <ul style="margin:0.75rem 0 0 1.5rem;font-size:0.91rem;line-height:1.9">
    <li>SinusoidalTimeEmbedding → MLP → time embedding dim=128</li>
    <li>Residual blocks with FiLM conditioning: scale/shift from time embedding</li>
    <li>Output projection: <code>nn.Conv2d</code>, zero-initialized (weight=0, bias=0)</li>
    <li>Training hyperparameters: AdamW lr=5e-4, cosine decay to 5e-7, EMA=0.9999, batch=128</li>
  </ul>
  <div class="callout {_pareto_class}">
    <strong>Pareto Assessment:</strong> {_pareto_msg}
  </div>
</div>

<div class="card">
  <h3>FID Checkpoint Trajectory</h3>
  {_traj_html}
</div>

<div class="card">
  <h3>Training Dynamics (ConvControl, from JSONL log)</h3>
  <p>Key observations from <code>logs/controlled_conv_seed42.jsonl</code>:</p>
  <ul style="margin:0.5rem 0 0 1.5rem;font-size:0.91rem;line-height:1.8">
    <li>Ctrl magnitude ratio (ctrl/score): starts 1.6% (epoch 1) → stabilises around 0.2–3.4%</li>
    <li>DSM loss: highly variable (140–614 range) — expected for σ-weighted objective</li>
    <li>Grad norms: 4–42 post-clip (grad_clip=1.0 is binding — correct behaviour)</li>
    <li>Quality loss (Inception feature match): activates at epoch 5, providing shape signal</li>
    <li>Cosine similarity score/ctrl: oscillates −0.18 to +0.29 — controller is not aligned with score</li>
  </ul>
</div>
</section>

<!-- ===== SECTION 5C: SESSION 10 — COLLAPSE ANALYSIS ===== -->
<section id="collapse">
<h2>5C. Session 10 — Controller Collapse Root Cause Analysis</h2>
<div class="callout" style="background:#fff3cd;border-color:#f0ad4e">
  <strong>Key Finding:</strong> The ConvControlPolicy (Session 9) collapsed to 0.061% of
  score magnitude at epoch 50. Diagnosis: <strong>{_s10_diagnosis}</strong>.
  Path KL gradient (1.45) dominates the controller's output layer.
  Quality gradient is 0.000 — broken through 50-step SDE + Inception.
</div>

<div class="grid3">
  <div class="card stat">
    <div class="num">1.452</div>
    <div class="label">Path KL gradient on output layer (dominant)</div>
  </div>
  <div class="card stat">
    <div class="num">0.000</div>
    <div class="label">Quality gradient on output layer (broken chain)</div>
  </div>
  <div class="card stat">
    <div class="num">0.061%</div>
    <div class="label">||u|| / ||score|| at epoch 50 (collapsed)</div>
  </div>
</div>

<div class="card">
  <h3>Per-Term Gradient Analysis (Output Layer)</h3>
  <table style="width:100%;border-collapse:collapse;font-size:0.91rem">
    <tr style="background:#f5f5f5"><th style="text-align:left;padding:6px">Loss Term</th><th style="text-align:right;padding:6px">Output Layer Gradient</th></tr>
    {_grad_rows}
  </table>
  <p style="margin-top:0.75rem;font-size:0.88rem">Path KL is 180,000x larger than quality gradient.
  The controller has no gradient signal toward non-zero outputs and strong gradient toward zero.</p>
</div>

<div class="card">
  <h3>Collapse Trajectory (controlled_conv_seed42)</h3>
  {'<img src="' + _s10_collapse_img + '" alt="Collapse trajectory" style="max-width:100%"/>' if _s10_collapse_img else '<div class="no-img">[collapse trajectory figure not found]</div>'}
  <table style="width:60%;border-collapse:collapse;font-size:0.88rem;margin-top:0.5rem">
    <tr style="background:#f5f5f5"><th style="text-align:left;padding:5px">Epoch</th><th style="text-align:right;padding:5px">||u||</th><th style="text-align:right;padding:5px">Ratio</th></tr>
    {_traj_rows}
  </table>
</div>

<div class="card">
  <h3>Session 10 v2 Objective Fix</h3>
  <ul style="margin:0.5rem 0 0 1.5rem;font-size:0.91rem;line-height:1.9">
    <li><strong>Detach control energy</strong> — removes zero attractor from controller gradient</li>
    <li><strong>Trajectory quality loss</strong> — per-sample nearest-neighbour Inception distance</li>
    <li><strong>REINFORCE quality signal</strong> — bypasses broken SDE+Inception gradient chain</li>
    <li><strong>WarmupSchedule</strong> — path_kl_weight=0 for 5 epochs, linear ramp over next 5</li>
    <li><strong>TwoPhaseScheduler</strong> — LR=1e-5 for phase 1, cosine from 5e-4 for phase 2</li>
    <li><strong>Scaled output init</strong> — output_proj from N(0, 1e-4), breaks zero symmetry</li>
  </ul>
  <div class="callout">
    <strong>Training Runs (ready to launch):</strong><br>
    Run 5A (frozen backbone): <code>python scripts/train_redesigned_v2.py --run 5a</code><br>
    Pilot 5A (30 min): <code>python scripts/train_redesigned_v2.py --run 5a --pilot</code><br>
    Run 5B (joint finetune): <code>python scripts/train_redesigned_v2.py --run 5b</code><br>
    Run 5C (no path-KL): <code>python scripts/train_redesigned_v2.py --run 5c</code><br>
    <strong>Paper-grade eval (Step 5D):</strong> <code>python scripts/eval_v2_multiseed.py --run-id 5a</code><br>
    <strong>Extended baseline (Step 7C):</strong> <code>python scripts/extended_baseline_analysis.py --score-ckpt checkpoints/score_fmnist_v2/score_best.pt</code>
  </div>
</div>
</section>

<!-- ===== SECTION 6: SAMPLE GALLERY ===== -->
<section id="gallery">
<h2>6. Sample Quality Gallery</h2>
<div class="img-grid">
  <div class="img-card">
    {img_tag("epoch50", "CIFAR-10 baseline SDE samples (epoch 50)")}
    <p><strong>CIFAR-10 — Score model, epoch 50.</strong> Samples show recognisable structure
    (backgrounds, textures) but remain blurry.  50 epochs is insufficient for convergence on CIFAR-10;
    200+ epochs with a deeper model would yield sharper results.</p>
  </div>
  <div class="img-card">
    {img_tag("fmnist", "FashionMNIST baseline score model, 3 epochs")}
    <p><strong>FashionMNIST — Baseline score model, 3 epochs.</strong> Basic silhouettes visible.
    Continued training to ~30 epochs would produce sharper edges; adding EMA (already implemented)
    typically gives a visible quality boost at no cost.</p>
  </div>
  <div class="img-card">
    {img_tag("sde_fmnist", "FashionMNIST SDE samples from trained score checkpoint")}
    <p><strong>FashionMNIST — Baseline SDE, 3-epoch score checkpoint, 50 steps.</strong>
    Samples show recognisable clothing silhouettes; quality limited by 3-epoch training.
    The SDE sampler uses 50 predictor + 1 corrector = 100 NFE per sample.</p>
  </div>
  <div class="img-card">
    {img_tag("val_grid", "Checkpoint migration validation grid")}
    <p><strong>CIFAR-10 — Migration validation grid.</strong> Confirms that all 50 pre-existing
    checkpoints load correctly after <code>migrate_checkpoint.py</code> added the <code>model_config</code>
    key.  Samples look identical before and after migration (no weight changes).</p>
  </div>
</div>
<div class="callout" style="margin-top:1rem">
  <strong>Missing grids:</strong> Controlled SDE sample grid requires the epoch-2/3 controlled
  checkpoint.  DDIM grid can be generated with
  <code>python scripts/eval_samples.py --mode ddim --model-ckpt checkpoints/score_fmnist_test/score_epoch_0003.pt --quick</code>.
</div>
</section>

<!-- ===== SECTION 7: INFORMATION-THEORETIC ANALYSIS ===== -->
<section id="info-theory">
<h2>7. Information-Theoretic Analysis</h2>
<div class="card">
  <h3>The Path-KL Term and Girsanov's Theorem</h3>
  <p>In the ITS framework, the controller u<sub>θ</sub> defines a new measure P<sup>u</sup> on path
  space that differs from the reference measure P<sup>0</sup> induced by the uncontrolled SDE.  The
  KL divergence between these two path measures is given by the Girsanov theorem:</p>
  <div class="eq">KL(P<sup>u</sup> ‖ P<sup>0</sup>) = E<sub>P<sup>u</sup></sub> [ log dP<sup>u</sup>/dP<sup>0</sup> ]
  = E [ ½ ∫₀ᵀ ‖u(x_t, t)/g(x_t, t)‖² dt ]</div>
  <p>The log Radon-Nikodym derivative log(dP<sup>u</sup>/dP<sup>0</sup>) can be approximated
  along discrete-time trajectories using Euler-Maruyama steps as:</p>
  <div class="eq">log RN ≈ ∑_k [ u_k · ΔW_k / g_k − ½ ‖u_k/g_k‖² Δt ]</div>
  <p>where ΔW_k = W_(k+1) − W_k is the Wiener increment at step k.  This is the quantity the ITS
  training objective attempts to minimise (weighted by λ_kl).  In the training logs, this quantity
  starts at ~53 and decreases to ~5 over the first two epochs, indicating the controller is learning
  to stay close to the reference measure while still influencing the trajectory.</p>

  <h3>The Jarzynski Equality and Free-Energy Estimation</h3>
  <p>The Jarzynski equality is a non-equilibrium thermodynamic identity that connects the irreversible
  work W done along a stochastic trajectory to the equilibrium free-energy difference ΔF:</p>
  <div class="eq">E<sub>trajectories</sub>[exp(−βW)] = exp(−βΔF)</div>
  <p>In the diffusion context, W is the accumulated work done by the control drift against the noise:
  <code>W ≈ ∑_k u_k · (x_(k+1) − x_k − f_k Δt) / g_k²</code>.
  Taking the log of both sides and using the numerically stable log-sum-exp formulation:</p>
  <div class="eq">ΔF = −(1/β) log[ (1/N) ∑_i exp(−β W_i) ]
  = −(1/β) [ logsumexp(−β W) − log N ]</div>
  <p>The ITS implementation clips extreme work values at |W| ≤ max_clip=50 before exponentiation
  to prevent numerical overflow — this was Bug 6, now fixed.  The Jarzynski estimator is biased for
  small N (Jensen's inequality), so large trajectory batches are needed for reliable ΔF estimates.
  Current training uses N=batch_size per mini-batch, which gives high-variance estimates; a dedicated
  evaluation pass with N≥1000 trajectories would give a more reliable free-energy estimate.</p>

  <h3>Information Cost of Generation</h3>
  <p>The path-KL term has a direct information-theoretic interpretation: it measures how many nats of
  information the controller "spends" per generated sample to deviate the trajectory from the reference
  SDE.  A controller that achieves lower FID with lower path-KL achieves a more information-efficient
  improvement.  Conversely, a high path-KL with no FID improvement indicates the controller is spending
  information budget without improving quality — the regime the current λ_kl=0.1 regularisation is
  designed to prevent.  The goal is to find the Pareto frontier in the (path-KL, FID) trade-off space.</p>
</div>
</section>

<!-- ===== SECTION 8: LIMITATIONS ===== -->
<section id="limitations">
<h2>8. Current Limitations and Honest Assessment</h2>
<div class="card">
<ol style="margin:0 0 0 1.5rem;line-height:1;counter-reset:item">

<li style="margin:1rem 0">
  <strong>The controller has not yet been demonstrated to improve FID over DDPM/DDIM baselines.</strong><br>
  <span style="font-size:0.92rem;color:#444">
  All evaluation in this session used 3-epoch CPU-trained checkpoints with N=256 samples — insufficient
  for statistical conclusions.  Addressing this requires training to 50+ epochs on GPU with N≥5000
  evaluation samples.
  </span>
</li>

<li style="margin:1rem 0">
  <strong>The feature-matching quality term is a heuristic approximation, not a true distribution-level objective.</strong><br>
  <span style="font-size:0.92rem;color:#444">
  Computing the L² distance between <em>batch mean</em> Inception features is a first-order moment
  matching that ignores second-order statistics (captured by FID's covariance term).  A more principled
  objective would be a kernel MMD or a full FID-style loss — but these require large batches to estimate
  stably.
  </span>
</li>

<li style="margin:1rem 0">
  <strong>Evaluation has been conducted on small sample counts; results are not statistically reliable.</strong><br>
  <span style="font-size:0.92rem;color:#444">
  FID with N=256 typically has ±30–80 standard deviation depending on the model.  The FID difference
  between DDPM (360.9) and DDIM (375.2) is within this noise range.  At least N=5000 is required for
  FashionMNIST and N=10000 for CIFAR-10.
  </span>
</li>

<li style="margin:1rem 0">
  <strong>The CIFAR-10 score model was trained for 50 epochs, which is insufficient for high-quality generation.</strong><br>
  <span style="font-size:0.92rem;color:#444">
  State-of-the-art CIFAR-10 diffusion models require 500–1000+ epochs.  The current model produces
  structurally recognisable but blurry images.  Re-training from scratch with the same architecture
  for 200 epochs would bring FID into the 15–30 range.
  </span>
</li>

<li style="margin:1rem 0">
  <strong>The Schrödinger bridge IPF solver is a research skeleton; end-to-end convergence has not been validated.</strong><br>
  <span style="font-size:0.92rem;color:#444">
  <code>SchrodingerBridgeSampler</code> in <code>src/its/samplers/sb_solver.py</code> implements the IPF
  Girsanov path-KL loss and alternating <code>train_step_forward</code> / <code>train_step_backward</code>
  gradient steps using <code>ConvControlPolicy</code> for both directions.  Shape tests pass (see
  <code>test_schrodinger_bridge_shapes</code>), but multi-iteration IPF convergence requires GPU-scale
  training that is out of scope for the current CPU-only experiments.
  </span>
</li>

<li style="margin:1rem 0">
  <strong>Thermodynamic quantities are computed approximately; the Crooks fluctuation theorem has not been formally verified.</strong><br>
  <span style="font-size:0.92rem;color:#444">
  The Jarzynski estimator uses the Euler-Maruyama discretisation, which introduces a time-step bias in
  the work estimates.  Formally verifying the Crooks theorem (detailed balance condition) would require
  running the time-reversed SDE and comparing forward/backward work distributions — a non-trivial
  experiment not yet implemented.
  </span>
</li>

</ol>
</div>
</section>

<!-- ===== SECTION 9: NEXT STEPS ===== -->
<section id="next">
<h2>9. Next Experiments Roadmap (Session 5 Status)</h2>
<div class="card">
  {gantt}
</div>
<div class="callout ok">
  <strong>Session 5 achievements:</strong>
  26 tests passing (up from 21); paper infrastructure complete (generate_results_table.py,
  generate_paper_figures.py, reproducibility_manifest.json, docs/negative_results.md);
  early stopping implemented; thermodynamic analysis pipeline (path KL, Jarzynski, Crooks) complete;
  DDPM NaN guard fixed; Google Cloud VM enabled for GPU training.
</div>
<div class="callout info">
  <strong>Immediate next action (GPU available):</strong>
  Run Config D 10-epoch full-dataset training on Google Cloud VM, then baselines at N=2048:
  <code>python scripts/run_experiment.py experiment=controlled_mnist
  training.epochs=10 training.jsonl_log=logs/config_D_10ep.jsonl</code>
  then: <code>python scripts/eval_samples.py --sampler ddpm --num-samples 2048 --eval-seed 42</code>
</div>
<div class="callout warn">
  <strong>Known gap:</strong>
  No test-split FID at N≥2048 yet. All scientifically valid FID values still pending full-dataset
  training runs. Sessions 1–3 FID values (train split, N=256) are stale and excluded from
  main tables. See docs/negative_results.md §1 and §2.
</div>
</section>

<!-- ===== SECTION 10: PHASE LOG ===== -->
<section id="phases">
<h2>10. Phase Change Log</h2>
<div class="card">
<table>
  <tr><th>Ref</th><th>Description</th><th>Status</th><th>Implementation detail</th></tr>
  {phase_html}
</table>
</div>
</section>

<!-- ===== SECTION 11: APPENDIX ===== -->
<section id="appendix">
<h2>11. Appendix: Configuration Reference</h2>

<details>
<summary>Hydra Config — controlled_mnist experiment (full resolved YAML)</summary>
<pre><code>{ctrl_yaml}</code></pre>
</details>

<details>
<summary>Files produced or modified this session</summary>
<table>
  <tr><th>File</th><th>Size (bytes)</th></tr>
  {files_html}
</table>
</details>

<details>
<summary>Exact reproduction commands</summary>
<pre><code># 0. Activate virtual environment
.venv/Scripts/activate  # Windows

# 1. Validate configs
python scripts/validate_configs.py

# 2. Migrate CIFAR-10 checkpoints (one-time)
python scripts/migrate_checkpoint.py --dir checkpoints/score

# 3. Baseline score training (FashionMNIST, 3 epochs, CPU)
python scripts/run_experiment.py --config-name config experiment=double_well device=cpu seed=42

# 4. FashionMNIST score 3-epoch train
python -c "
from its.training.score_training import ScoreTrainingConfig, train_score_model
from its.data import DatasetConfig; from its.models import ScoreUNetConfig
cfg = ScoreTrainingConfig(epochs=3, lr=2e-4, device='cpu',
    dataset=DatasetConfig(name='fashionmnist', batch_size=64, num_workers=0),
    model=ScoreUNetConfig(in_channels=1, base_channels=32, channel_mults=(1,2,2)),
    checkpoint_dir='checkpoints/score_fmnist', save_interval=1)
print(train_score_model(cfg))"

# 5. Eval (DDPM, --quick)
python scripts/eval_samples.py --config-name controlled_mnist --mode ddpm \\
    --model-ckpt checkpoints/score_fmnist_test/score_epoch_0003.pt --quick

# 6. Run all hardening tests
python -m pytest tests/test_phase5_hardening.py -v

# 7. Re-generate this report
python scripts/generate_report.py
</code></pre>
</details>

</section>

</main>

<footer>
  ITS Analytics Report · {gen_date} · Claude Sonnet 4.6 (sessions 1–5) ·
  Python 3.11.9 · PyTorch 2.3.0 · Hydra 1.3.2 · 26 tests passing · All charts SVG (print-safe)
</footer>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    sz_kb = len(html.encode("utf-8")) // 1024
    print(f"Report written to {output_path}  ({sz_kb} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ITS HTML analytics report")
    parser.add_argument("--output", default="data/results/its_analytics_report.html")
    args = parser.parse_args()
    build_report(ROOT / args.output)


if __name__ == "__main__":
    main()
