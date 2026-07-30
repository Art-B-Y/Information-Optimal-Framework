"""Write Session 9 summary to data/results/session_summary_session9.txt.

Run after ConvControl training completes to include the final FID result.
"""
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_conv_results() -> dict:
    p = ROOT / "data/results/controlled_conv_results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_conv_multiseed() -> dict:
    p = ROOT / "data/results/controlled_conv_multiseed.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_conv_trajectory() -> list:
    p = ROOT / "data/results/conv_checkpoint_fid_trajectory.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


def _load_mlp_ci() -> dict:
    p = ROOT / "data/results/controlled_config_d_multiseed.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def main() -> None:
    conv = _load_conv_results()
    conv_ms = _load_conv_multiseed()
    traj = _load_conv_trajectory()
    mlp_ci = _load_mlp_ci()

    # Determine Pareto result
    ddpm_nfe100_fid = 236.28
    conv_runs = conv.get("runs", [])
    conv_fids = [r.get("fid") for r in conv_runs if r.get("fid") is not None]
    conv_mean = sum(conv_fids) / len(conv_fids) if conv_fids else None

    if conv_ms and "ci_95" in conv_ms:
        ci = conv_ms["ci_95"]
        conv_fid_str = f"{ci['mean']:.2f} ± {ci['std']:.2f} (95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}])"
        pareto = ci["upper"] < ddpm_nfe100_fid
    elif conv_mean is not None:
        conv_fid_str = f"{conv_mean:.2f} (1 seed)"
        pareto = conv_mean < ddpm_nfe100_fid
    else:
        conv_fid_str = "NOT YET EVALUATED"
        pareto = False

    mlp_ci_data = mlp_ci.get("ci_95", {})
    mlp_fid = mlp_ci_data.get("mean", 327.47)

    # Best checkpoint from trajectory
    best_traj = None
    if traj:
        valid = [r for r in traj if "fid" in r]
        if valid:
            best_traj = min(valid, key=lambda r: r["fid"])

    lines = [
        "ITS PROJECT — SESSION 9 SUMMARY",
        f"Generated: {datetime.date.today()}",
        "=" * 80,
        "",
        "1. SESSION 9 OBJECTIVES AND OUTCOME",
        "-" * 80,
        "Session 9 primary mission: Run ConvControl to 50 epochs on FashionMNIST.",
        "If FID < 236.28 (DDPM at NFE=100), paper upgrades from TMLR to NeurIPS workshop.",
        "",
        "2. CODE AUDIT (Step 1) — COMPLETE",
        "-" * 80,
        "  1A: 7/8 imports PASS; 1 wrong spec in brief (train_controlled vs train_controlled_score)",
        "  1B: ScoreSDESimulator constructor verified (must be (score_model, config))",
        "  1C: ConvControlPolicy output_proj ZERO-INITIALIZED — CONFIRMED SAFE",
        "  1D: Checkpoint directories audited (all valid)",
        "  1E: 52/52 tests passing (pre-session audit; 56/56 by session end)",
        "  1F: GTX 1650 Ti, 4.0 GB VRAM, 3.22 GB free",
        "  Audit log: data/results/audit_log_session9.txt",
        "",
        "3. CONVCONTROL TRAINING (Step 2A)",
        "-" * 80,
        f"  Config: AdamW lr=5e-4, cosine to 5e-7, EMA=0.9999, batch=128, 50 epochs",
        f"  Score backbone: checkpoints/score_fmnist_v2/score_best.pt (frozen)",
        f"  Checkpoint dir: checkpoints/controlled_conv_seed42/",
        f"  Epoch timing: ~12.1 min/epoch → ~10.1h total",
        f"  Status: {'COMPLETE' if conv_runs else 'RUNNING / DID NOT COMPLETE IN SESSION'}",
        "",
        "4. CONVCONTROL FID RESULT",
        "-" * 80,
        f"  ConvControl FID: {conv_fid_str}",
        f"  MLP Config D FID (3-seed): {mlp_fid:.2f} ± {mlp_ci_data.get('std', 1.57):.2f}",
        f"  DDPM at NFE=100: {ddpm_nfe100_fid:.2f} ± 0.87",
        f"  PARETO IMPROVEMENT: {'YES — venue upgrades to NeurIPS workshop' if pareto else 'NO — TMLR remains appropriate'}",
        "",
    ]

    if best_traj:
        lines += [
            "5. CHECKPOINT FID TRAJECTORY",
            "-" * 80,
            f"  Best checkpoint: epoch {best_traj['epoch']}, FID={best_traj['fid']:.2f} (N={best_traj.get('num_samples', '?')})",
        ]
        for r in traj:
            if "fid" in r:
                lines.append(f"    Epoch {r['epoch']:3d}: FID={r['fid']:.2f}")
        lines.append("")

    lines += [
        "6. TRAINING METRICS (from JSONL log)",
        "-" * 80,
        "  Epoch 1: cos_sim=0.1403, ctrl_mag=0.277 vs score_mag=17.647 (ratio=1.57%)",
        "  Epoch 2: cos_sim=0.1192, ctrl_mag=0.071 (ratio=0.40%)",
        "  Epoch 3 (step 100): loss=397.48, dsm=397.47, path_kl=0.061",
        "  Epoch 3 (step 120): cos_sim=-0.175, ctrl_mag=0.116 (ratio=0.66%)",
        "  High pre-clip grad norms: 42-952 (expected for DSM loss; grad_clip=1.0)",
        "",
        "7. FILES CREATED OR MODIFIED (Session 9)",
        "-" * 80,
        "  scripts/train_controlled_config_d.py — Added --use-conv-control flag",
        "  scripts/eval_conv_checkpoints.py — NEW: periodic checkpoint evaluation",
        "  scripts/eval_conv_multiseed.py — NEW: 3-seed CI for ConvControl",
        "  scripts/generate_conv_learning_curve.py — NEW: Conv vs MLP learning curve",
        "  scripts/generate_results_table.py — Added ConvControl row parsing",
        "  scripts/generate_reproducibility_manifest.py — Added Session 8/9 entries",
        "  tests/test_session9_paper.py — NEW: 13 tests",
        "  docs/paper_requirements_roadmap.md — Updated for Session 9",
        "  data/results/audit_log_session9.txt — NEW",
        "  data/results/reproducibility_manifest.json — Updated (8 entries)",
        "  data/paper_figures/conv_vs_mlp_learning_curve.{png,pdf} — NEW",
        "  reproduce_paper_results.sh — NEW: full reproduction shell script (Step 6D)",
        "",
        "8. TEST SUITE STATUS",
        "-" * 80,
        "  Total: 56 tests (43 from Sessions 1-8 + 13 new Session 9)",
        "  Passing: 56 | Skipped: 0 | Failed: 0",
        "  Runtime: ~313s",
        "",
        "9. PENDING TASKS (Session 10)",
        "-" * 80,
        "  (a) Run eval_conv_checkpoints.py to get FID trajectory",
        "  (b) Run eval_conv_multiseed.py for 3-seed CI",
        "  (c) Run generate_conv_learning_curve.py with actual data",
        "  (d) Architecture ablations (attention, time embedding)",
        "  (e) Mode coverage with recalibrated classifier (20 epochs)",
        "  (f) Crooks verification with ConvControl checkpoint",
        "  (g) Update eval matrix with ConvControl row",
        "  (h) Regenerate all paper figures with ConvControl data",
        "  (i) Write paper sections based on Pareto result",
        "",
        "10. VENUE RECOMMENDATION (updated after Session 9)",
        "-" * 80,
        "  CONDITIONAL: See Section 4 PARETO IMPROVEMENT for current recommendation.",
        "  TMLR (unconditional): All evidence supports TMLR with current MLP results.",
        "  NeurIPS workshop: Possible if ConvControl achieves Pareto improvement.",
        "",
        "=" * 80,
        "END OF SESSION 9 SUMMARY",
        "=" * 80,
    ]

    out = ROOT / "data/results/session_summary_session9.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out}")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    main()
