"""IPF Alternating Training Driver for Schrödinger Bridge.

Implements the Iterative Proportional Fitting (IPF) alternating training loop:
  For each IPF iteration:
    Phase "forward":  freeze backward_policy, train forward_policy for --epochs-per-phase
    Phase "backward": freeze forward_policy, train backward_policy for --epochs-per-phase

JSONL log records include: ipf_iteration, phase, global_step, batch_idx,
    ipf_loss, ctrl_norm, grad_norm, wall_clock_seconds.

Usage:
    # Quick smoke test (2 IPF iterations, 1 epoch/phase, 500 sample subset)
    python scripts/train_schrodinger_bridge.py \\
        --ipf-iterations 2 --epochs-per-phase 1 --subset 500 \\
        --checkpoint-dir checkpoints/ipf_fmnist \\
        --jsonl-log logs/ipf_fmnist.jsonl

    # With evaluation (saves 16-sample PNG grids)
    python scripts/train_schrodinger_bridge.py \\
        --ipf-iterations 5 --epochs-per-phase 2 \\
        --checkpoint-dir checkpoints/ipf_fmnist \\
        --evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from its.samplers import SBSamplerConfig, SchrodingerBridgeSampler


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_fashionmnist(batch_size: int, subset: Optional[int] = None, seed: int = 42) -> DataLoader:
    from torchvision import datasets, transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST(root="./data/tensors", train=True, download=True, transform=tfm)
    if subset is not None and subset < len(ds):
        import random
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), subset)
        ds = Subset(ds, indices)
    _nw = min(4, max(0, (os.cpu_count() or 1) - 1))
    _dl_kwargs: dict = {"batch_size": batch_size, "shuffle": True,
                        "num_workers": _nw, "pin_memory": True}
    if _nw > 0:
        _dl_kwargs["prefetch_factor"] = 2
        _dl_kwargs["persistent_workers"] = True
    return DataLoader(ds, **_dl_kwargs)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

class _JsonlWriter:
    def __init__(self, path: Optional[str]) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_ipf_checkpoint(
    path: Path,
    sampler: SchrodingerBridgeSampler,
    fwd_optim: torch.optim.Optimizer,
    bwd_optim: torch.optim.Optimizer,
    ipf_iter: int,
    global_step: int,
) -> None:
    torch.save({
        "ipf_iteration": ipf_iter,
        "epoch": ipf_iter,  # alias for segmented launcher epoch tracking
        "global_step": global_step,
        "forward_policy_state_dict": sampler.forward_policy.state_dict(),
        "backward_policy_state_dict": sampler.backward_policy.state_dict(),
        "fwd_optimizer_state_dict": fwd_optim.state_dict(),
        "bwd_optimizer_state_dict": bwd_optim.state_dict(),
        "sampler_config": {
            "dt": sampler.config.dt,
            "steps": sampler.config.steps,
            "diffusion": sampler.config.diffusion,
            "in_channels": sampler.config.in_channels,
            "base_channels": sampler.config.base_channels,
            "num_blocks": sampler.config.num_blocks,
            "time_embed_dim": sampler.config.time_embed_dim,
        },
    }, path)


# ---------------------------------------------------------------------------
# Evaluation / visualisation
# ---------------------------------------------------------------------------

def _save_sample_grid(
    sampler: SchrodingerBridgeSampler,
    save_path: Path,
    ipf_iter: int,
    device: torch.device,
    n: int = 16,
    channels: int = 1,
    height: int = 28,
    width: int = 28,
) -> None:
    """Generate n forward samples and save as a PNG grid."""
    try:
        import torchvision.utils as vutils
    except ImportError:
        logging.getLogger(__name__).warning("torchvision not available; skipping grid save.")
        return

    sampler.eval()
    with torch.no_grad():
        x0 = torch.randn(n, channels, height, width, device=device)
        result = sampler.sample_forward(x0, use_policy=True)
        gen = result["x_T"].clamp(-1, 1)

    # Denormalise from [-1,1] to [0,1]
    gen = (gen + 1.0) * 0.5
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(gen, str(save_path), nrow=4, padding=2)
    logging.getLogger(__name__).info("Saved sample grid to %s", save_path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_ipf(
    ipf_iterations: int,
    epochs_per_phase: int,
    batch_size: int,
    subset: Optional[int],
    checkpoint_dir: str,
    jsonl_log: Optional[str],
    evaluate: bool,
    lr: float,
    device_str: str,
    sampler_cfg: SBSamplerConfig,
    nan_tolerance: int,
    log_interval: int,
    seed: int,
    resume_latest: bool = False,
    max_wall_hours: float = 0.0,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    torch.manual_seed(seed)
    device = torch.device(device_str)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    jsonl_writer = _JsonlWriter(jsonl_log)
    loader = _load_fashionmnist(batch_size, subset=subset, seed=seed)

    sampler = SchrodingerBridgeSampler(sampler_cfg).to(device)

    # output_proj is zero-initialised for training stability.  Re-initialise
    # with small nonzero weights so the IPF path-KL loss is nonzero from step 1
    # and gradients can flow.  This is the standard approach for image IPF
    # experiments; pure zero-start leads to vanishing gradients indefinitely.
    import torch.nn as nn
    for policy in (sampler.forward_policy, sampler.backward_policy):
        nn.init.normal_(policy.output_proj.weight, std=0.01)
        nn.init.zeros_(policy.output_proj.bias)

    fwd_optim = torch.optim.Adam(sampler.forward_policy.parameters(), lr=lr)
    bwd_optim = torch.optim.Adam(sampler.backward_policy.parameters(), lr=lr)

    global_step = 0
    t_start = time.time()
    last_fwd_loss = float("nan")
    last_bwd_loss = float("nan")
    start_ipf_iter = 1

    # --resume-latest: load ipf_last.pt if it exists.
    if resume_latest:
        last_ckpt = ckpt_dir / "ipf_last.pt"
        if last_ckpt.exists():
            try:
                resume_state = torch.load(last_ckpt, map_location=device, weights_only=False)
                start_ipf_iter = int(resume_state.get("ipf_iteration", 0)) + 1
                sampler.forward_policy.load_state_dict(
                    resume_state["forward_policy_state_dict"])
                sampler.backward_policy.load_state_dict(
                    resume_state["backward_policy_state_dict"])
                fwd_optim.load_state_dict(resume_state["fwd_optimizer_state_dict"])
                bwd_optim.load_state_dict(resume_state["bwd_optimizer_state_dict"])
                global_step = int(resume_state.get("global_step", 0))
                logger.info("Resumed from %s at IPF iteration %d", last_ckpt, start_ipf_iter - 1)
            except Exception as _e:
                logger.warning("Could not resume from %s: %s — starting fresh.", last_ckpt, _e)

    for ipf_iter in range(start_ipf_iter, ipf_iterations + 1):
        logger.info("=== IPF iteration %d / %d ===", ipf_iter, ipf_iterations)

        for phase, optim, freeze_other in [
            ("forward",  fwd_optim, sampler.backward_policy),
            ("backward", bwd_optim, sampler.forward_policy),
        ]:
            logger.info("  Phase: %s", phase)

            # Freeze the 'other' policy for this phase.
            for p in freeze_other.parameters():
                p.requires_grad_(False)
            active_policy = (
                sampler.forward_policy if phase == "forward" else sampler.backward_policy
            )
            for p in active_policy.parameters():
                p.requires_grad_(True)

            nan_streak = 0

            for epoch in range(1, epochs_per_phase + 1):
                for batch_idx, (x_real, _) in enumerate(loader, start=1):
                    x_real = x_real.to(device)
                    global_step += 1

                    if phase == "forward":
                        metrics = sampler.train_step_forward(x_real, fwd_optim)
                        ipf_loss = metrics.get("loss_forward", float("nan"))
                    else:
                        metrics = sampler.train_step_backward(x_real, bwd_optim)
                        ipf_loss = metrics.get("loss_backward", float("nan"))

                    # NaN guard.
                    if not (ipf_loss == ipf_loss) or ipf_loss == float("inf"):
                        nan_streak += 1
                        logger.warning(
                            "NaN/Inf IPF loss at iter=%d phase=%s step=%d (streak=%d)",
                            ipf_iter, phase, global_step, nan_streak,
                        )
                        if nan_streak >= nan_tolerance:
                            raise RuntimeError(
                                f"IPF training collapsed: {nan_streak} consecutive NaN "
                                f"batches at iteration={ipf_iter} phase={phase}."
                            )
                        continue
                    nan_streak = 0

                    if phase == "forward":
                        last_fwd_loss = ipf_loss
                    else:
                        last_bwd_loss = ipf_loss

                    if log_interval and global_step % log_interval == 0:
                        logger.info(
                            "iter=%d phase=%s epoch=%d step=%d ipf_loss=%.4f ctrl_norm=%.4f grad_norm=%.4f",
                            ipf_iter, phase, epoch, global_step, ipf_loss,
                            metrics.get("ctrl_norm_forward", metrics.get("ctrl_norm_backward", 0.0)),
                            metrics.get("grad_norm", 0.0),
                        )
                        record = {
                            "ipf_iteration": ipf_iter,
                            "phase": phase,
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch_idx": batch_idx,
                            "ipf_loss": ipf_loss,
                            "ctrl_norm": float(metrics.get(
                                "ctrl_norm_forward", metrics.get("ctrl_norm_backward", 0.0)
                            )),
                            "grad_norm": float(metrics.get("grad_norm", 0.0)),
                            "wall_clock_seconds": round(time.time() - t_start, 2),
                        }
                        jsonl_writer.write(record)

            # Unfreeze after phase completes.
            for p in freeze_other.parameters():
                p.requires_grad_(True)

        # Save checkpoint at end of each IPF iteration.
        ckpt_path = ckpt_dir / f"ipf_iter_{ipf_iter:04d}.pt"
        _save_ipf_checkpoint(ckpt_path, sampler, fwd_optim, bwd_optim, ipf_iter, global_step)
        logger.info("Saved checkpoint: %s", ckpt_path)

        # Save latest pointer.
        latest = ckpt_dir / "ipf_last.pt"
        import shutil
        shutil.copy2(str(ckpt_path), str(latest))

        # Optional evaluation grid — saved to data/results/ per paper spec.
        if evaluate:
            results_dir = _ROOT / "data" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            grid_path = results_dir / f"sb_samples_iter_{ipf_iter}_fmnist.png"
            _save_sample_grid(sampler, grid_path, ipf_iter, device, n=16,
                              channels=sampler_cfg.in_channels)

        # Log IPF iteration summary.
        summary = {
            "ipf_iteration": ipf_iter,
            "phase": "summary",
            "global_step": global_step,
            "last_fwd_loss": last_fwd_loss,
            "last_bwd_loss": last_bwd_loss,
            "wall_clock_seconds": round(time.time() - t_start, 2),
        }
        jsonl_writer.write(summary)
        logger.info(
            "IPF iter %d complete: fwd_loss=%.4f bwd_loss=%.4f elapsed=%.1fs",
            ipf_iter, last_fwd_loss, last_bwd_loss, time.time() - t_start,
        )

        # Wall-clock limit: exit with code 42 for the segmented launcher.
        if max_wall_hours > 0 and (time.time() - t_start) > max_wall_hours * 3600:
            logger.info(
                "Wall-clock limit %.2f h reached after IPF iter %d; exiting 42.",
                max_wall_hours, ipf_iter,
            )
            import sys as _sys
            _sys.exit(42)

    return {
        "ipf_iterations_completed": ipf_iterations,
        "global_steps": global_step,
        "last_fwd_loss": last_fwd_loss,
        "last_bwd_loss": last_bwd_loss,
        "wall_clock_sec": round(time.time() - t_start, 2),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IPF alternating training for Schrödinger Bridge.")
    p.add_argument("--ipf-iterations", type=int, default=5, help="Number of IPF half-iteration pairs.")
    p.add_argument("--epochs-per-phase", type=int, default=2, help="Epochs per forward/backward phase.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--subset", type=int, default=None, help="Limit dataset to N samples (for fast tests).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/ipf_fmnist")
    p.add_argument("--jsonl-log", type=str, default=None, help="Path for JSONL step log.")
    p.add_argument("--evaluate", action="store_true", default=False, help="Save 16-sample grids after each IPF iter.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--nan-tolerance", type=int, default=5)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run", default=None,
                   help="Run ID passed by the segmented launcher (ignored by this script).")
    p.add_argument("--resume-latest", action="store_true",
                   help="Resume from ipf_last.pt in checkpoint dir (used by segmented launcher).")
    p.add_argument("--max-wall-hours", type=float, default=0.0,
                   help="Exit with code 42 after this many wall-clock hours (used by segmented launcher).")
    # SBSamplerConfig overrides
    p.add_argument("--steps", type=int, default=20, help="Euler-Maruyama steps per trajectory.")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--diffusion", type=float, default=1.0)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--time-embed-dim", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    jsonl_log = args.jsonl_log
    if jsonl_log is None:
        log_dir = Path("logs") / "ipf"
        log_dir.mkdir(parents=True, exist_ok=True)
        jsonl_log = str(log_dir / f"ipf_{int(time.time())}.jsonl")

    sampler_cfg = SBSamplerConfig(
        dt=args.dt,
        steps=args.steps,
        diffusion=args.diffusion,
        in_channels=1,          # FashionMNIST
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        time_embed_dim=args.time_embed_dim,
    )

    results = train_ipf(
        ipf_iterations=args.ipf_iterations,
        epochs_per_phase=args.epochs_per_phase,
        batch_size=args.batch_size,
        subset=args.subset,
        checkpoint_dir=args.checkpoint_dir,
        jsonl_log=jsonl_log,
        evaluate=args.evaluate,
        lr=args.lr,
        device_str=device_str,
        sampler_cfg=sampler_cfg,
        nan_tolerance=args.nan_tolerance,
        log_interval=args.log_interval,
        seed=args.seed,
        resume_latest=args.resume_latest,
        max_wall_hours=args.max_wall_hours,
    )
    print("IPF training complete:", results)
