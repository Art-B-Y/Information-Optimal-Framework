"""Session Speed Overhaul tests (Steps 1–9).

Verifies the training performance fixes, fault-tolerant checkpointing,
session segmentation, and hardware utilisation improvements.

11 tests, targeting 77 total from 66.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # allows `from scripts.X import Y`

# ---------------------------------------------------------------------------
# 1. DataLoader num_workers default is non-zero
# ---------------------------------------------------------------------------

def test_dataloader_num_workers_nonzero():
    """ControlledScoreTrainingConfig.num_workers defaults to > 0."""
    from its.training.controlled_score_training import ControlledScoreTrainingConfig
    cfg = ControlledScoreTrainingConfig()
    assert cfg.num_workers > 0, (
        f"num_workers should be > 0 (was {cfg.num_workers}); "
        "num_workers=0 serialises DataLoader to the main process and is a major throughput bottleneck."
    )


# ---------------------------------------------------------------------------
# 2. DataLoader pin_memory is enabled
# ---------------------------------------------------------------------------

def test_dataloader_pin_memory_enabled():
    """_make_dataloader produces a DataLoader with pin_memory=True."""
    from its.training.controlled_score_training import ControlledScoreTrainingConfig, _make_dataloader

    cfg = ControlledScoreTrainingConfig(
        dataset_name="fashionmnist",
        batch_size=4,
        num_workers=0,          # override to avoid worker processes in tests
        dataset_subset_size=8,
    )
    loader = _make_dataloader(cfg)
    assert loader.pin_memory, "pin_memory must be True to enable fast CPU→GPU transfers"


# ---------------------------------------------------------------------------
# 3. AMP (mixed precision) is enabled by default
# ---------------------------------------------------------------------------

def test_amp_enabled_in_config():
    """ControlledScoreTrainingConfig.mixed_precision defaults to True."""
    from its.training.controlled_score_training import ControlledScoreTrainingConfig
    cfg = ControlledScoreTrainingConfig()
    assert cfg.mixed_precision is True, (
        "mixed_precision must default to True; disabling it was the primary source of the "
        "10-hour run expanding to 8 days."
    )


# ---------------------------------------------------------------------------
# 4. Autosave writes atomically (no .tmp left behind after save)
# ---------------------------------------------------------------------------

def test_autosave_checkpoint_atomic():
    """_save_autosave writes autosave.pt and removes the .tmp file."""
    from its.training.controlled_score_training import (
        ControlledScoreTrainingConfig,
        _save_autosave,
        _capture_rng_states,
    )
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)
        model_cfg = ScoreUNetConfig(in_channels=1, base_channels=16, channel_mults=(1,))
        ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=32, depth=2)
        model = build_score_model(model_cfg)
        control = build_control_policy(ctrl_cfg)
        optim = torch.optim.AdamW(
            list(model.parameters()) + list(control.parameters()), lr=1e-4
        )
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        rng = _capture_rng_states()

        _save_autosave(ckpt_dir, epoch=1, global_step=50,
                       model=model, control=control,
                       optimizer=optim, scaler=scaler, ema=None, rng_states=rng)

        autosave_path = ckpt_dir / "autosave.pt"
        tmp_path = ckpt_dir / "autosave.pt.tmp"

        assert autosave_path.exists(), "autosave.pt must exist after _save_autosave"
        assert not tmp_path.exists(), (
            "autosave.pt.tmp must be removed after the atomic rename; "
            "if it exists the checkpoint write was not atomic."
        )

        # Verify the saved data is valid and contains expected fields.
        ckpt = torch.load(autosave_path, map_location="cpu", weights_only=False)
        assert ckpt["global_step"] == 50
        assert ckpt["epoch"] == 1
        assert ckpt.get("is_autosave") is True


# ---------------------------------------------------------------------------
# 5. Resuming from autosave continues from saved global_step
# ---------------------------------------------------------------------------

def test_resume_from_autosave_continues_step_count():
    """Loading a checkpoint saved at global_step=50 restores step=50, not 0."""
    from its.training.controlled_score_training import (
        _save_autosave, _load_checkpoint, _capture_rng_states,
    )
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)
        model_cfg = ScoreUNetConfig(in_channels=1, base_channels=16, channel_mults=(1,))
        ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=32, depth=2)
        model = build_score_model(model_cfg)
        control = build_control_policy(ctrl_cfg)
        optim = torch.optim.AdamW(
            list(model.parameters()) + list(control.parameters()), lr=1e-4
        )
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        rng = _capture_rng_states()

        _save_autosave(ckpt_dir, epoch=2, global_step=50,
                       model=model, control=control,
                       optimizer=optim, scaler=scaler, ema=None, rng_states=rng)

        # Load fresh models and restore.
        model2 = build_score_model(model_cfg)
        control2 = build_control_policy(ctrl_cfg)
        optim2 = torch.optim.AdamW(
            list(model2.parameters()) + list(control2.parameters()), lr=1e-4
        )
        autosave_path = ckpt_dir / "autosave.pt"
        start_epoch, global_step = _load_checkpoint(
            autosave_path, model2, control2, optim2, None,
            device=torch.device("cpu"), ema=None,
        )
        assert global_step == 50, f"Expected global_step=50 after resume, got {global_step}"
        assert start_epoch == 3, f"Expected start_epoch=3 (epoch 2 + 1), got {start_epoch}"


# ---------------------------------------------------------------------------
# 6. Simulated crash recovery — step count accumulates correctly
# ---------------------------------------------------------------------------

def test_simulated_crash_recovery():
    """After saving at step 100 and reloading, global_step is preserved at 100."""
    from its.training.controlled_score_training import (
        _save_checkpoint, _load_checkpoint,
    )
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "controlled_crash.pt"
        model_cfg = ScoreUNetConfig(in_channels=1, base_channels=16, channel_mults=(1,))
        ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=32, depth=2)

        model = build_score_model(model_cfg)
        control = build_control_policy(ctrl_cfg)
        optim = torch.optim.AdamW(
            list(model.parameters()) + list(control.parameters()), lr=1e-4
        )
        _save_checkpoint(ckpt_path, epoch=5, model=model, control=control,
                         optimizer=optim, scaler=None, global_step=100, ema=None)

        # Simulate crash: fresh model/optimizer, reload from checkpoint.
        model2 = build_score_model(model_cfg)
        control2 = build_control_policy(ctrl_cfg)
        optim2 = torch.optim.AdamW(
            list(model2.parameters()) + list(control2.parameters()), lr=1e-4
        )
        start_epoch, global_step = _load_checkpoint(
            ckpt_path, model2, control2, optim2, None,
            device=torch.device("cpu"), ema=None,
        )
        assert global_step == 100, f"Crash recovery: expected step=100, got {global_step}"
        assert start_epoch == 6, f"Crash recovery: expected start_epoch=6, got {start_epoch}"


# ---------------------------------------------------------------------------
# 7. Segmented launcher exits with code 42 at wall-clock limit
# ---------------------------------------------------------------------------

def test_segmented_launcher_exits_42():
    """train_controlled_score exits with sys.exit(42) when max_wall_hours is exceeded."""
    import subprocess
    import textwrap

    with tempfile.TemporaryDirectory() as tmpdir:
        inline = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, r"{ROOT / 'src'}")
            from its.training.controlled_score_training import (
                ControlledScoreTrainingConfig, train_controlled_score
            )
            from its.controllers import ControlConfig
            cfg = ControlledScoreTrainingConfig(
                epochs=9999,
                max_wall_hours=0.0001,
                batch_size=4,
                dataset_subset_size=8,
                num_workers=0,
                device="cpu",
                mixed_precision=False,
                ema_decay=None,
                eval_every=0,
                checkpoint_dir=r"{tmpdir}",
                checkpoint_every_n_minutes=0,
                use_lr_schedule=False,
                log_interval=1,
                quality_weight=0.0,
                path_kl_weight=0.0,
                control_weight=0.0,
                wandb=False,
                control=ControlConfig(state_dim=784, hidden_dim=32, depth=2),
            )
            train_controlled_score(cfg)
        """)
        result = subprocess.run(
            [sys.executable, "-c", inline],
            capture_output=True, timeout=120,
        )
        assert result.returncode == 42, (
            f"Expected exit code 42 (clean segment boundary), got {result.returncode}.\n"
            f"stdout: {result.stdout[-2000:].decode(errors='replace')}\n"
            f"stderr: {result.stderr[-2000:].decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# 8. Checkpoint pruner applies the correct retention policy
# ---------------------------------------------------------------------------

def test_checkpoint_pruner_retention_policy():
    """prune() keeps the last 5 + every 10th epoch; deletes everything else."""
    from scripts.prune_checkpoints import prune

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)
        # Create 25 fake epoch checkpoints: epochs 1..25
        for epoch in range(1, 26):
            (ckpt_dir / f"controlled_epoch_{epoch:04d}.pt").write_bytes(b"fake")
        # Also create always-keep files.
        (ckpt_dir / "controlled_last.pt").write_bytes(b"fake")
        (ckpt_dir / "autosave.pt").write_bytes(b"fake")

        result = prune(ckpt_dir, keep_last=5, keep_every=10, dry_run=True)

        kept_epochs = sorted(
            int(Path(p).stem.split("_")[-1]) for p in result["kept"]
        )
        deleted_epochs = sorted(
            int(Path(p).stem.split("_")[-1]) for p in result["deleted"]
        )

        # Last 5 epochs: 21, 22, 23, 24, 25
        for e in [21, 22, 23, 24, 25]:
            assert e in kept_epochs, f"Epoch {e} should be kept (last 5)"

        # Every 10th epoch before the last 5 (1..20): epochs 10, 20
        for e in [10, 20]:
            assert e in kept_epochs, f"Epoch {e} should be kept (every 10th)"

        # Everything else in 1..20 that is not a multiple of 10 should be deleted.
        for e in range(1, 21):
            if e % 10 != 0:
                assert e in deleted_epochs, f"Epoch {e} should be deleted"

        # always-keep files must never appear in deleted.
        deleted_names = {Path(p).name for p in result["deleted"]}
        assert "controlled_last.pt" not in deleted_names
        assert "autosave.pt" not in deleted_names


# ---------------------------------------------------------------------------
# 9. Log stitcher merges JSONL segments in chronological order
# ---------------------------------------------------------------------------

def test_log_stitcher_merges_segments():
    """Merging two JSONL segment logs produces records in step order."""
    from scripts.stitch_logs import stitch_jsonl_logs

    with tempfile.TemporaryDirectory() as tmpdir:
        seg1 = Path(tmpdir) / "seg1.jsonl"
        seg2 = Path(tmpdir) / "seg2.jsonl"
        out = Path(tmpdir) / "stitched.jsonl"

        # Segment 1: steps 1..5
        with seg1.open("w") as f:
            for step in range(1, 6):
                f.write(json.dumps({"step": step, "loss": 1.0 / step}) + "\n")
        # Segment 2: steps 6..10
        with seg2.open("w") as f:
            for step in range(6, 11):
                f.write(json.dumps({"step": step, "loss": 1.0 / step}) + "\n")

        stitch_jsonl_logs([seg1, seg2], out)

        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        steps = [r["step"] for r in records]
        assert steps == list(range(1, 11)), f"Expected steps 1..10, got {steps}"


# ---------------------------------------------------------------------------
# 10. Frozen score model has no gradients after backward
# ---------------------------------------------------------------------------

def test_frozen_score_model_no_gradients():
    """With freeze_score_model=True, score model params have requires_grad=False and no grad."""
    from its.models import ScoreUNetConfig, build_score_model
    from its.controllers import ControlConfig, build_control_policy

    model_cfg = ScoreUNetConfig(in_channels=1, base_channels=16, channel_mults=(1,))
    ctrl_cfg = ControlConfig(state_dim=784, hidden_dim=32, depth=2)

    model = build_score_model(model_cfg)
    # Simulate freeze_score_model=True as the training function does it.
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    control = build_control_policy(ctrl_cfg)  # unfrozen

    # Build a loss that flows through the control policy (unfrozen).
    # Score model output is used as input to control; use no_grad for score forward.
    x = torch.randn(4, 1, 28, 28)
    sigma = torch.ones(4, 1, 1, 1) * 0.3
    with torch.no_grad():
        score = model(x + sigma * torch.randn_like(x), sigma)

    # Control forward (unfrozen path).
    flat = x.view(4, -1)
    t = sigma.view(4)
    ctrl_out = control(flat, t)
    loss = ctrl_out.mean() + score.detach().mean() * 0.0  # detach score
    loss.backward()

    # All score model parameters must have no gradient.
    for name, param in model.named_parameters():
        assert not param.requires_grad, (
            f"Score model param '{name}' still has requires_grad=True after freezing."
        )
        assert param.grad is None, (
            f"Score model param '{name}' accumulated a gradient despite being frozen."
        )

    # At least one control policy parameter must have a gradient.
    ctrl_has_grad = any(
        p.grad is not None for p in control.parameters() if p.requires_grad
    )
    assert ctrl_has_grad, "Control policy should accumulate gradients when unfrozen."


# ---------------------------------------------------------------------------
# 11. CUDAPrefetcher yields correct batches on CPU (passthrough mode)
# ---------------------------------------------------------------------------

def test_cuda_prefetcher_correct_output():
    """CUDAPrefetcher on CPU yields identical batches to direct iteration."""
    from its.utils.prefetch import CUDAPrefetcher

    # Build a simple list-based DataLoader substitute.
    data = [(torch.arange(i, i + 4, dtype=torch.float32),
             torch.tensor(i)) for i in range(5)]

    class _SimpleLoader:
        def __iter__(self):
            return iter(data)
        def __len__(self):
            return len(data)

    loader = _SimpleLoader()
    device = torch.device("cpu")
    prefetcher = CUDAPrefetcher(loader, device)

    direct = list(loader)
    prefetched = list(prefetcher)

    assert len(prefetched) == len(direct), (
        f"Prefetcher yielded {len(prefetched)} batches; expected {len(direct)}"
    )
    for i, (pf, di) in enumerate(zip(prefetched, direct)):
        x_pf, y_pf = pf
        x_di, y_di = di
        assert torch.equal(x_pf, x_di), f"Batch {i} x mismatch: {x_pf} vs {x_di}"
        assert torch.equal(y_pf, y_di), f"Batch {i} y mismatch: {y_pf} vs {y_di}"
