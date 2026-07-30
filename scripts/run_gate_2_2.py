"""Phase 2 Step 2 — Gate 2.2: the baseline validation gate.

This is the decisive checkpoint of the project. No ITS number means anything until the
UNCONTROLLED baseline is proven to work. Its absence is what let ten sessions build a
thermodynamic interpretation on top of a sampler that emitted noise.

Five criteria, ALL of which must pass:
  1. Stable score-model training (loss decreases, no NaN/divergence, bounded grad norm,
     val tracks train)
  2. FID decreases during training
  3. FID improves with larger NFE     <-- the criterion whose failure was the clearest
                                          symptom of the broken sampler (flat FID ~326)
  4. Baseline reaches a competitive FID at paper grade (5000 samples, 3 seeds)
  5. Samples are visually recognisable

Writes data/results/gate_2_2_report.json and docs/gate_2_2_report.md.

Usage:
    python scripts/run_gate_2_2.py
    python scripts/run_gate_2_2.py --quick        # smaller N, for a fast dry run (NOT a result)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_NAME = "fmnist_score_corrected_baseline"
SIGMA_MIN, SIGMA_MAX, SIGMA_DATA = 0.01, 42.0, 0.70
SEEDS = [42, 123, 7]
PAPER_N = 5000
FID_TARGET = 30.0


def _load_model(ckpt_path: Path, base_channels: int, device, use_ema: bool = False):
    """Load a checkpoint. Defaults to RAW weights, NOT EMA -- deliberately.

    Phase 2 finding: the EMA in this run had no bias correction and was seeded with the
    RANDOM INITIALISATION, so at decay=0.9999 the weight still on the init is decay^t --
    80.0% at epoch 10 and 10.8% at epoch 100. Measured at epoch 10:
        RAW weights : FID  99.75
        EMA weights : FID 373.07
    i.e. the "EMA" checkpoint was mostly noise. ExponentialMovingAverage has since been
    fixed (warmup schedule), but the checkpoints ALREADY ON DISK carry the contaminated
    shadow, and the un-contaminated average cannot be recovered from them post hoc
    (it would require theta_init, which is not saved). So this run's gate is evaluated
    on raw weights, which are simply the trained model and are perfectly valid.
    Pass use_ema=True only to reproduce the contaminated numbers.

    The checkpoint stores `model_config` (a ScoreUNetConfig asdict), so the
    architecture is reconstructed from the file rather than guessed -- if the run used
    a different base_channels or preconditioning than this script's defaults, we still
    load it correctly instead of silently mismatching.
    EMA is stored as {"shadow": [tensors]} over params with requires_grad.
    """
    import torch
    from its.models import ScoreUNetConfig, build_score_model

    state = torch.load(ckpt_path, map_location=device)
    mc = state.get("model_config")
    if mc:
        mc = dict(mc)
        mc["channel_mults"] = tuple(mc.get("channel_mults", (1, 2, 2)))
        cfg = ScoreUNetConfig(**mc)
    else:
        cfg = ScoreUNetConfig(in_channels=1, base_channels=base_channels, channel_mults=(1, 2, 2),
                              use_time_embedding=True, time_embed_dim=128,
                              preconditioning="edm", sigma_data=SIGMA_DATA)
    m = build_score_model(cfg).to(device)
    m.load_state_dict(state.get("model_state_dict") or state["model"])

    if use_ema:
        ema = state.get("ema_state_dict") or state.get("ema")
        if ema and ema.get("shadow"):
            trainable = [p for p in m.parameters() if p.requires_grad]
            for p, q in zip(trainable, ema["shadow"]):
                p.data.copy_(q.to(device))
    m.eval()
    return m


def _plot(xs, ys, xlabel, ylabel, title, stem, logx=False):
    """Save a publication-quality figure to data/paper_figures/{stem}.{png,pdf}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
    ax.plot(xs, ys, "o-", color="#2b6cb0", linewidth=2, markersize=6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color="#444")
    if logx:
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "data" / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"   figure -> {out / (stem + '.png')}")


def _fid(model, nfe, num_samples, seed, device):
    from its.sde.score_sde import ScoreSDEConfig
    from its.eval.evaluator import evaluate_sampler, EvaluationConfig
    sde = ScoreSDEConfig(num_steps=nfe, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
                         clamp=0.0, parameterisation="ve")
    ev = EvaluationConfig(dataset_name="fashionmnist", num_samples=num_samples,
                          batch_size=250, device=str(device), seed=seed)
    return evaluate_sampler(model, None, sde, ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--base-channels", type=int, default=64)
    args = ap.parse_args()

    import torch
    device = torch.device("cuda")
    ckpt_dir = ROOT / "checkpoints" / RUN_NAME
    log_path = ROOT / "logs" / f"{RUN_NAME}.jsonl"
    n_traj = 1024 if args.quick else 2048
    n_paper = 1024 if args.quick else PAPER_N
    seeds = [42] if args.quick else SEEDS

    report = {"run": RUN_NAME, "quick_mode": args.quick, "criteria": {}}

    # ---------------- Criterion 1: stable training ----------------
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    # score_training.py writes step records with "loss"/"avg_loss"/"grad_norm", and
    # per-epoch validation records with "val_dsm_loss"/"train_dsm_loss"/"overfit_ratio".
    losses = [r["avg_loss"] for r in rows if isinstance(r.get("avg_loss"), (int, float))]
    gnorms = [r["grad_norm"] for r in rows if isinstance(r.get("grad_norm"), (int, float))]
    vals = [(r["epoch"], r["val_dsm_loss"]) for r in rows
            if isinstance(r.get("val_dsm_loss"), (int, float))]
    overfit = [r["overfit_ratio"] for r in rows if isinstance(r.get("overfit_ratio"), (int, float))]
    finite = all(math.isfinite(v) for v in losses)
    first_q = sum(losses[: max(1, len(losses)//4)]) / max(1, len(losses)//4)
    last_q = sum(losses[-max(1, len(losses)//4):]) / max(1, len(losses)//4)
    decreasing = last_q < first_q
    gn_bounded = (max(gnorms) < 1e3) if gnorms else None
    # val must track train without diverging: finite, decreasing, and never badly
    # above train (overfit_ratio > 1.2 is score_training's own warning threshold).
    val_ok = True
    if vals:
        val_ok = (math.isfinite(vals[-1][1]) and vals[-1][1] <= vals[0][1]
                  and (not overfit or max(overfit[-5:]) < 1.2))
    c1 = bool(finite and decreasing and (gn_bounded is not False) and val_ok)
    report["criteria"]["1_stable_training"] = {
        "pass": c1, "all_finite": finite,
        "loss_first_quarter_mean": first_q, "loss_last_quarter_mean": last_q,
        "decreasing": decreasing,
        "grad_norm_max": max(gnorms) if gnorms else None, "grad_norm_bounded": gn_bounded,
        "val_first": vals[0] if vals else None, "val_last": vals[-1] if vals else None,
        "val_tracks_train": val_ok,
        "overfit_ratio_last": overfit[-1] if overfit else None,
        "overfit_ratio_max": max(overfit) if overfit else None,
        "val_trajectory": vals,
    }
    print(f"C1 stable training      : {'PASS' if c1 else 'FAIL'}  "
          f"loss {first_q:.4f} -> {last_q:.4f}, grad_max={max(gnorms) if gnorms else 'n/a'}")

    # ---------------- Criterion 2: FID decreases during training ----------------
    ckpts = sorted(ckpt_dir.glob("score_epoch_*.pt"))
    picks = [c for c in ckpts if int(c.stem.split("_")[-1]) in (10, 30, 50, 70, 100)] or ckpts[-5:]
    traj = []
    for c in picks:
        ep = int(c.stem.split("_")[-1])
        m = _load_model(c, args.base_channels, device)
        f = _fid(m, 100, n_traj, 42, device)["fid"]
        traj.append({"epoch": ep, "fid": f, "num_samples": n_traj, "seed": 42})
        print(f"   epoch {ep:>3}: FID={f:.2f}  (N={n_traj}, seed 42)", flush=True)
    c2 = len(traj) >= 2 and traj[-1]["fid"] < traj[0]["fid"]
    report["criteria"]["2_fid_decreases_with_training"] = {"pass": bool(c2), "trajectory": traj}
    print(f"C2 FID vs training      : {'PASS' if c2 else 'FAIL'}")
    _plot([r["epoch"] for r in traj], [r["fid"] for r in traj],
          "Training epoch", "FID", "Corrected baseline: FID improves with training",
          "baseline_fid_vs_epoch")

    # ---------------- Criterion 3: FID improves with NFE ----------------
    best = min(traj, key=lambda r: r["fid"]) if traj else None
    best_ck = ckpt_dir / f"score_epoch_{best['epoch']:04d}.pt"
    m = _load_model(best_ck, args.base_channels, device)
    nfes = [50, 100, 200] if args.quick else [50, 100, 200, 500]
    nfe_rows = []
    for n in nfes:
        f = _fid(m, n, n_traj, 42, device)["fid"]
        nfe_rows.append({"nfe": n, "fid": f, "num_samples": n_traj, "seed": 42})
        print(f"   NFE {n:>3}: FID={f:.2f}", flush=True)
    c3 = nfe_rows[-1]["fid"] < nfe_rows[0]["fid"]
    report["criteria"]["3_fid_improves_with_nfe"] = {
        "pass": bool(c3), "best_epoch": best["epoch"], "curve": nfe_rows,
        "note": "The void pre-audit results showed FID flat at ~326 across all NFE, "
                "which is impossible for a working sampler. This is the criterion that "
                "most directly detects the broken sampler.",
    }
    print(f"C3 FID vs NFE           : {'PASS' if c3 else 'FAIL'}")
    _plot([r["nfe"] for r in nfe_rows], [r["fid"] for r in nfe_rows],
          "NFE (sampling steps)", "FID",
          "Corrected baseline: FID improves with NFE\n(the void results were flat at ~326)",
          "baseline_fid_vs_nfe", logx=True)

    # ---------------- Criterion 4: paper-grade FID ----------------
    best_nfe = min(nfe_rows, key=lambda r: r["fid"])["nfe"]
    runs = []
    for s in seeds:
        f = _fid(m, best_nfe, n_paper, s, device)["fid"]
        runs.append({"seed": s, "fid": f, "num_samples": n_paper, "nfe": best_nfe})
        print(f"   seed {s}: FID={f:.2f}  (N={n_paper}, NFE={best_nfe})", flush=True)
    mean = sum(r["fid"] for r in runs) / len(runs)
    std = (sum((r["fid"] - mean) ** 2 for r in runs) / max(1, len(runs) - 1)) ** 0.5 if len(runs) > 1 else 0.0
    c4 = mean < FID_TARGET
    report["criteria"]["4_competitive_fid"] = {
        "pass": bool(c4), "fid_mean": mean, "fid_std": std, "target": FID_TARGET,
        "num_samples": n_paper, "seeds": seeds, "nfe": best_nfe, "runs": runs,
        "paper_grade": (n_paper >= PAPER_N and len(seeds) >= 3),
    }
    print(f"C4 competitive FID      : {'PASS' if c4 else 'FAIL'}  {mean:.2f} +/- {std:.2f} (target <{FID_TARGET})")

    # ---------------- Criterion 5: visual ----------------
    import torchvision
    from its.sde.score_sde import ScoreSDEConfig, ScoreSDESimulator
    torch.manual_seed(42)
    sde = ScoreSDEConfig(num_steps=best_nfe, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
                         clamp=0.0, parameterisation="ve")
    x = ScoreSDESimulator(m, sde).sample((128, 1, 28, 28), device)
    grid_path = ROOT / "data" / "results" / "baseline_final_samples.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    # Data is normalised to [-1,1]; map back to [0,1] for viewing.
    torchvision.utils.save_image(x.cpu().clamp(-1, 1) * 0.5 + 0.5, str(grid_path), nrow=16)
    stats = {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
             "std": float(x.std()), "per_image_std_mean": float(x.std(dim=(1, 2, 3)).mean())}
    report["criteria"]["5_visual"] = {
        "pass": None, "grid": str(grid_path), "sample_stats": stats,
        "note": "Requires human/model inspection of the grid; auto-stats only indicate "
                "the samples are not degenerate (constant or pure noise).",
    }
    print(f"C5 visual               : grid -> {grid_path}  stats={stats}")

    report["gate_verdict"] = {
        "pass": bool(c1 and c2 and c3 and c4),
        "note": "Criterion 5 requires visual inspection; recorded separately.",
    }
    out = ROOT / "data" / "results" / "gate_2_2_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nGate verdict (C1-C4): {'PASS' if report['gate_verdict']['pass'] else 'FAIL'}")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
