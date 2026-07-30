"""Session 9 paper-readiness tests.

Tests:
- test_score_sde_simulator_constructor_requires_model
- test_conv_control_output_zero_initialized
- test_conv_control_forward_shape
- test_conv_control_checkpoint_loadable
- test_train_controlled_config_d_has_conv_control_flag
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Step 1B: ScoreSDESimulator constructor
# ---------------------------------------------------------------------------

def test_score_sde_simulator_constructor_requires_model():
    """ScoreSDESimulator must be called as ScoreSDESimulator(score_model, config)."""
    from its.sde.score_sde import ScoreSDESimulator
    from its.sde import ScoreSDEConfig
    from its.models import ScoreUNetConfig, build_score_model

    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=32, channel_mults=(1, 2),
                                use_time_embedding=True, time_embed_dim=64)
    model = build_score_model(model_cfg)
    sde_cfg = ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=10)

    # Correct usage must not raise
    sim = ScoreSDESimulator(model, sde_cfg)
    assert sim is not None

    # Wrong usage (bare config as first arg) must raise TypeError
    with pytest.raises(TypeError):
        ScoreSDESimulator(sde_cfg)


# ---------------------------------------------------------------------------
# Step 1C: ConvControlPolicy architecture
# ---------------------------------------------------------------------------

def test_conv_control_output_zero_initialized():
    """ConvControlPolicy output_proj must start zero-initialized."""
    from its.controllers.neural_control import ConvControlPolicy, ConvControlConfig

    cfg = ConvControlConfig(in_channels=1, base_channels=32, num_blocks=2, time_embed_dim=64)
    policy = ConvControlPolicy(cfg)

    assert torch.all(policy.output_proj.weight == 0), "output_proj.weight not zero"
    assert torch.all(policy.output_proj.bias == 0), "output_proj.bias not zero"


def test_conv_control_forward_shape():
    """ConvControlPolicy must output the same shape as its input."""
    from its.controllers.neural_control import ConvControlPolicy, ConvControlConfig

    cfg = ConvControlConfig(in_channels=1, base_channels=32, num_blocks=2, time_embed_dim=64)
    policy = ConvControlPolicy(cfg)
    policy.eval()

    B, C, H, W = 4, 1, 28, 28
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)
    with torch.no_grad():
        out = policy(x, t)

    assert out.shape == (B, C, H, W), f"Expected {(B,C,H,W)}, got {out.shape}"


def test_conv_control_checkpoint_loadable():
    """ablation_conv/controlled_last.pt must load into ConvControlPolicy without error."""
    ckpt_path = ROOT / "checkpoints" / "ablation_conv" / "controlled_last.pt"
    if not ckpt_path.exists():
        pytest.skip("ablation_conv checkpoint not found")

    from its.controllers.neural_control import ConvControlPolicy, ConvControlConfig

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ctrl_cfg_dict = state.get("control_config", {})
    assert "base_channels" in ctrl_cfg_dict, "Expected ConvControlConfig in checkpoint"

    cfg = ConvControlConfig(
        in_channels=ctrl_cfg_dict.get("in_channels", 1),
        base_channels=ctrl_cfg_dict.get("base_channels", 64),
        num_blocks=ctrl_cfg_dict.get("num_blocks", 3),
        time_embed_dim=ctrl_cfg_dict.get("time_embed_dim", 128),
        dropout=ctrl_cfg_dict.get("dropout", 0.0),
    )
    policy = ConvControlPolicy(cfg)
    policy.load_state_dict(state["control_state_dict"], strict=True)
    assert state["epoch"] == 5, f"Expected epoch 5, got {state['epoch']}"


def test_train_controlled_config_d_has_conv_control_flag():
    """train_controlled_config_d.py must accept --use-conv-control CLI flag."""
    import subprocess
    result = subprocess.run(
        ["python", str(ROOT / "scripts" / "train_controlled_config_d.py"), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert "--use-conv-control" in result.stdout, (
        "--use-conv-control flag missing from train_controlled_config_d.py help"
    )


def test_generate_results_table_parses_conv_results(tmp_path):
    """generate_results_table.py must parse controlled_conv_results.json."""
    import json, sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_results_table import _extract_rows

    fake = {"config": "D_conv", "epochs": 50, "runs": [
        {"seed": 42, "fid": 220.0, "num_samples_eval": 2048},
    ]}
    (tmp_path / "controlled_conv_results.json").write_text(json.dumps(fake))

    baselines, proposed = _extract_rows(tmp_path)
    conv_rows = [r for r in proposed if "ConvControl" in r.get("sampler", "")]
    assert len(conv_rows) >= 1, "No ConvControl row found in proposed rows"
    assert conv_rows[0]["fid"] == pytest.approx(220.0, abs=0.1)
    assert conv_rows[0]["epochs"] == 50


def test_eval_conv_checkpoints_script_importable():
    """eval_conv_checkpoints.py must import without error."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_conv_checkpoints",
        str(ROOT / "scripts" / "eval_conv_checkpoints.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "eval_checkpoint")
    assert hasattr(mod, "main")


def test_eval_conv_multiseed_script_importable():
    """eval_conv_multiseed.py must import without error."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_conv_multiseed",
        str(ROOT / "scripts" / "eval_conv_multiseed.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "ci_95")
    assert hasattr(mod, "eval_one")

    # ci_95 must handle single value
    result = mod.ci_95([300.0])
    assert result["mean"] == pytest.approx(300.0)
    assert result["std"] == pytest.approx(0.0)


def test_conv_control_run_one_seed_uses_correct_hyperparameters():
    """ConvControl training must use lr=5e-4, EMA=0.9999, batch=128 (not MLP defaults)."""
    import importlib.util, inspect
    spec = importlib.util.spec_from_file_location(
        "train_ctrl",
        str(ROOT / "scripts" / "train_controlled_config_d.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    src = inspect.getsource(mod) if hasattr(mod, "__file__") else ""

    source = (ROOT / "scripts" / "train_controlled_config_d.py").read_text()

    # Verify ConvControl branch uses the expected hyperparameters
    assert "lr = 5e-4" in source, "ConvControl lr must be 5e-4"
    assert "lr_min = 5e-7" in source, "ConvControl lr_min must be 5e-7"
    assert "ema_decay = 0.9999" in source, "ConvControl EMA must be 0.9999"
    assert "batch_size = 128" in source, "ConvControl batch_size must be 128"


# ---------------------------------------------------------------------------
# Step 9 brief-specified tests (additional)
# ---------------------------------------------------------------------------

def test_convcontrol_film_conditioning_bounded():
    """FiLM scale and shift must be finite and bounded for t=0 and t=1."""
    from its.controllers.neural_control import ConvControlPolicy, ConvControlConfig, _FiLMLayer

    cfg = ConvControlConfig(in_channels=1, base_channels=32, num_blocks=2, time_embed_dim=64)
    policy = ConvControlPolicy(cfg)
    policy.eval()

    for t_val in [0.0, 1.0]:
        t = torch.full((4,), t_val)
        with torch.no_grad():
            t_emb = policy.time_embed(t)  # (4, 64)
            # Check FiLM layer of first block
            film = policy.blocks[0].film
            x_dummy = torch.randn(4, 32, 7, 7)
            scale = film.scale_proj(t_emb).view(4, -1, 1, 1) + 1.0
            shift = film.shift_proj(t_emb).view(4, -1, 1, 1)

        assert torch.all(torch.isfinite(scale)), f"scale not finite at t={t_val}"
        assert torch.all(torch.isfinite(shift)), f"shift not finite at t={t_val}"
        assert float(scale.abs().max()) <= 10.0, f"scale too large ({float(scale.abs().max()):.2f}) at t={t_val}"
        assert float(shift.abs().max()) <= 10.0, f"shift too large ({float(shift.abs().max()):.2f}) at t={t_val}"


def test_select_best_checkpoint_under_3_epochs(tmp_path):
    """select_best_checkpoint.py must handle a log with only 2 epochs without crashing."""
    import json, importlib.util
    spec = importlib.util.spec_from_file_location(
        "select_best",
        str(ROOT / "scripts" / "select_best_checkpoint.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Write synthetic 2-epoch JSONL log
    records = [
        {"epoch": 1, "global_step": 40, "dsm_loss": 400.0, "path_kl": 0.08, "control_energy": 0.001},
        {"epoch": 2, "global_step": 80, "dsm_loss": 350.0, "path_kl": 0.05, "control_energy": 0.002},
    ]
    jsonl_path = tmp_path / "test_log.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records))

    summaries = mod._load_epoch_summaries(jsonl_path)
    assert len(summaries) == 2, f"Expected 2 epoch summaries, got {len(summaries)}"

    # select_best takes (jsonl_path, checkpoint_dir, dsm_tolerance)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    best = mod.select_best(jsonl_path, ckpt_dir, dsm_tolerance=0.05)
    # Returns None (no checkpoint files), but must not crash on 2-epoch log
    # The function logs to stdout but handles missing files gracefully


def test_crooks_crossing_point_within_range():
    """crooks_crossing_point must return a value within 0.15 of the analytical crossing."""
    import numpy as np
    from its.physics.fluctuation import crooks_crossing_point

    rng = np.random.default_rng(123)
    # crooks_crossing_point compares hist(fwd_work) vs hist(-rev_work).
    # fwd ~ N(2.0, 0.5), rev ~ N(-1.0, 0.5) → -rev ~ N(1.0, 0.5)
    # The two distributions overlap at 1σ separation; crossing of N(2.0,0.5)
    # and N(1.0,0.5) occurs at midpoint = 1.5, in a high-density region.
    fwd_work = rng.normal(loc=2.0, scale=0.5, size=2000).tolist()
    rev_work = rng.normal(loc=-1.0, scale=0.5, size=2000).tolist()

    crossing = crooks_crossing_point(fwd_work, rev_work)
    analytical_crossing = 1.5

    assert abs(crossing - analytical_crossing) <= 0.15, (
        f"Crossing {crossing:.4f} deviates more than 0.15 from analytical value {analytical_crossing}"
    )


def test_paper_requirements_roadmap_sections():
    """docs/paper_requirements_roadmap.md must contain all 8 required sections."""
    roadmap = ROOT / "docs" / "paper_requirements_roadmap.md"
    assert roadmap.exists(), "paper_requirements_roadmap.md not found"

    content = roadmap.read_text(encoding="utf-8")
    required_sections = [
        "Section 1",
        "Section 2",
        "Section 3",
        "Section 4",
        "Section 5",
        "Section 6",
        "Section 7",
        "Section 8",
    ]
    for section in required_sections:
        assert section in content, f"Missing '{section}' in paper_requirements_roadmap.md"
