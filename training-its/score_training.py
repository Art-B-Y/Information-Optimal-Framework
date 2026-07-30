from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

try:  # pragma: no cover - optional dependency
    from torchmetrics import MeanMetric  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class MeanMetric:  # fall-back tracker
        def __init__(self) -> None:
            self.total = 0.0
            self.count = 0

        def to(self, device: torch.device) -> "MeanMetric":
            return self

        def update(self, value: torch.Tensor) -> None:
            self.total += float(value.item())
            self.count += 1

        def compute(self) -> torch.Tensor:
            if self.count == 0:
                return torch.tensor(0.0)
            return torch.tensor(self.total / self.count)

        def reset(self) -> None:
            self.total = 0.0
            self.count = 0

from ..data import DatasetConfig, build_dataset
from ..models import ScoreUNetConfig, build_score_model
from ..utils import configure_logging

logger = logging.getLogger(__name__)


def _sample_sigma(batch: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    log_min = torch.log(torch.tensor(sigma_min))
    log_max = torch.log(torch.tensor(sigma_max))
    u = torch.rand(batch.shape[0], device=batch.device)
    sigma = torch.exp(log_min + u * (log_max - log_min))
    return sigma.view(-1, *([1] * (batch.dim() - 1)))


@dataclass
class ScoreTrainingConfig:
    epochs: int = 10
    lr: float = 2e-4
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    grad_clip: float = 1.0
    device: Optional[str] = None
    log_interval: int = 100
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ScoreUNetConfig = field(default_factory=ScoreUNetConfig)
    ema_decay: Optional[float] = 0.999
    checkpoint_dir: Optional[str] = None
    save_interval: int = 0
    resume_from: Optional[str] = None
    log_dir: Optional[str] = None
    run_name: Optional[str] = None
    log_file: Optional[str] = None
    nan_tolerance: int = 5
    jsonl_log: Optional[str] = None
    # Step 7A: cosine annealing LR schedule (lr_min=0 disables)
    lr_min: float = 1e-6
    use_lr_schedule: bool = True
    # Speed overhaul: AMP enabled by default when CUDA is available.
    mixed_precision: bool = True
    # Step 7B: validation loss every N epochs (0 disables)
    val_every_n_epochs: int = 5
    # Step 7C: sample diversity (disabled by default; set to True to log)
    log_sample_diversity: bool = False
    # Segmented training: exit with code 42 after this many wall-clock hours (0 = disabled).
    max_wall_hours: float = 0.0
    # Phase 2 (2026-07-15): DSM loss weighting lambda(sigma).
    #
    # "none" reproduces the pre-Phase-2 objective  mean((score - (-eps/sigma))^2).
    # That is UNWEIGHTED: the target magnitude is 1/sigma, so the loss scale goes as
    # 1/sigma^2 -- a 1e4 ratio across sigma in [0.01, 1.0], and 1.7e7 across the
    # corrected range [0.01, 42].  The gradient is therefore dominated almost entirely
    # by the smallest sigmas, and the model barely learns the large-sigma regime --
    # which is exactly where the reverse trajectory STARTS.  This is an independent
    # defect from the four fixed in Phase 1.
    #
    # "sigma_sq" applies the standard lambda(sigma) = sigma^2:
    #     sigma^2 * (score + eps/sigma)^2 = (sigma*score + eps)^2
    # i.e. the objective becomes equivalent to eps-prediction, whose loss scale is
    # O(1) at every sigma.  This is the standard choice (Ho et al.; Song et al.;
    # Karras et al.) and is required for the corrected [0.01, 42] range to train.
    dsm_weighting: str = "none"


class ExponentialMovingAverage:
    """EMA of model parameters, with warmup so the random init cannot contaminate it.

    Phase 2 (2026-07-15) fix. The previous implementation seeded ``shadow`` with the
    RANDOM INITIALISATION and then decayed it at a fixed rate with no bias correction:

        shadow_t = decay^t * theta_init + (1 - decay) * sum_i decay^(t-i) * theta_i

    so the weight remaining on the random init is decay^t.  At the configured
    decay=0.9999 over this project's ~223 steps/epoch that is:

        epoch  10 ->  80.0% of the EMA weights are still the RANDOM INIT
        epoch  50 ->  32.8%
        epoch 100 ->  10.8%

    Measured consequence: the epoch-10 EMA checkpoint evaluated at FID 375 while the
    same configuration with raw weights reached FID ~75 -- the "EMA" checkpoint was
    mostly noise.  Any FID-vs-epoch curve built on these weights measures how fast the
    EMA forgets its initialisation, not how fast the model learns.

    Fix: use the standard warmup schedule (Adam-style bias correction expressed as an
    effective decay), which starts the average at the first real parameter value and
    anneals to ``decay``:

        decay_t = min(decay, (1 + t) / (10 + t))

    At t=0 this is 0.1, so the shadow is immediately dominated by real weights; it
    approaches ``decay`` as t grows.  This is what makes decay=0.9999 usable on a
    ~22k-step run.  Set ``use_warmup=False`` to reproduce the old (broken) behaviour.
    """

    def __init__(self, model: nn.Module, decay: float, use_warmup: bool = True) -> None:
        self.decay = decay
        self.use_warmup = use_warmup
        self.num_updates = 0
        self.shadow = [param.detach().clone() for param in model.parameters() if param.requires_grad]

    def _effective_decay(self) -> float:
        if not self.use_warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self._effective_decay()
        self.num_updates += 1
        for shadow_param, param in zip(self.shadow, model.parameters()):
            if not param.requires_grad:
                continue
            shadow_param.mul_(d).add_(param.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for shadow_param, param in zip(self.shadow, model.parameters()):
            if not param.requires_grad:
                continue
            param.data.copy_(shadow_param)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": [param.detach().clone() for param in self.shadow],
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.decay = state.get("decay", self.decay)
        self.num_updates = int(state.get("num_updates", 0))
        for target, source in zip(self.shadow, state.get("shadow", [])):
            target.copy_(source)


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    global_step: int,
) -> None:
    import dataclasses

    payload: dict[str, object] = {
        "epoch": epoch,
        "global_step": global_step,
        # Use canonical key names. Legacy aliases kept for backward-compat loaders.
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # Legacy aliases so old load code (checkpoint["model"]) still works.
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
        payload["ema"] = ema.state_dict()  # legacy alias
    # Save model architecture config so it can be reloaded without knowing the
    # original construction parameters (fixes architecture mismatch on load).
    if hasattr(model, "config"):
        payload["model_config"] = dataclasses.asdict(model.config)
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    logger.info("Resumed from %s (epoch=%d, step=%d)", path, start_epoch - 1, global_step)
    return start_epoch, global_step


def _resolve_log_file(config: ScoreTrainingConfig) -> Optional[str]:
    if config.log_file:
        return config.log_file
    if not config.log_dir:
        return None
    name = config.run_name or "score_train"
    stamp = torch.randint(0, 1_000_000, (1,)).item()
    return str(Path(config.log_dir) / f"{name}_{stamp}.log")


def train_score_model(config: ScoreTrainingConfig) -> dict[str, float]:
    """Train a score model with denoising score matching (DSM) and return final loss statistics."""
    log_file = _resolve_log_file(config)
    configure_logging(log_file=log_file)
    resolved_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(resolved_device, str) and resolved_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        resolved_device = "cpu"
    device = torch.device(resolved_device)

    # ------------------------------------------------------------------
    # Convention guard (Phase 2). The VE/VP mismatch -- training with the VE kernel
    # `x + sigma*eps` while sampling with VP dynamics -- was one of the four fatal
    # defects that voided every result this project ever produced. It must never
    # recur silently, so state the convention explicitly at every run and refuse
    # configurations that are known-invalid for it.
    # ------------------------------------------------------------------
    logger.info(
        "DIFFUSION CONVENTION: VE (kernel x + sigma*eps, DSM target -eps/sigma), "
        "sigma in [%.4g, %.4g], weighting=%s, preconditioning=%s. "
        "Sampling MUST use ScoreSDEConfig(parameterisation='ve') with the SAME sigma range.",
        config.sigma_min, config.sigma_max, config.dsm_weighting,
        getattr(config.model, "preconditioning", "legacy"),
    )
    if config.sigma_max <= 1.0:
        logger.warning(
            "sigma_max=%.4g is too small for a valid VE prior. Song's technique 1 sets "
            "sigma_max to the max pairwise data distance (measured 41.88 for "
            "FashionMNIST in [-1,1]); at sigma_max<=1 the perturbed distribution still "
            "retains most data structure, so the sampler's N(0, sigma_max^2) init is "
            "NOT the prior. See data/results/phase2_preflight.txt Finding A.",
            config.sigma_max,
        )
    if config.sigma_max / max(config.sigma_min, 1e-12) > 100 and config.dsm_weighting == "none":
        raise ValueError(
            f"dsm_weighting='none' with sigma range [{config.sigma_min}, {config.sigma_max}] "
            f"(ratio {config.sigma_max/config.sigma_min:.0f}x) is unweighted DSM: the loss "
            f"scale spans {(config.sigma_max/config.sigma_min)**2:.3g}x and the gradient "
            "collapses onto the smallest sigmas. Use dsm_weighting='sigma_sq'. "
            "See data/results/phase2_preflight.txt Finding C."
        )

    loaders = build_dataset(config.dataset)
    train_loader = loaders["train"]
    val_loader = loaders.get("val") or loaders.get("test")
    model = build_score_model(config.model).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    ema = ExponentialMovingAverage(model, config.ema_decay) if config.ema_decay else None

    _use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)

    # Step 7A: cosine annealing LR scheduler over total training steps.
    total_steps = config.epochs * len(train_loader)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps), eta_min=config.lr_min
        )
        if config.use_lr_schedule else None
    )

    loss_metric = MeanMetric().to(device)
    epoch_losses: list[float] = []

    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0
    if config.resume_from:
        ckpt_path = Path(config.resume_from)
        if ckpt_path.is_file():
            start_epoch, global_step = _load_checkpoint(ckpt_path, model, optimizer, ema, device)
        else:
            logger.warning("Checkpoint %s not found; starting from scratch.", ckpt_path)

    # JSONL log writer.
    _jsonl_path = Path(config.jsonl_log) if config.jsonl_log else None
    if _jsonl_path is not None:
        _jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_jsonl(record: dict) -> None:
        if _jsonl_path is None:
            return
        with _jsonl_path.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(record) + "\n")

    train_start = time.time()
    nan_streak = 0

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        loss_metric.reset()
        for batch in train_loader:
            x, _ = batch
            x = x.to(device)
            sigma = _sample_sigma(x, config.sigma_min, config.sigma_max)
            noise = torch.randn_like(x)
            noisy = x + sigma * noise
            target = -noise / sigma
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                score = model(noisy, sigma)
                # See ScoreTrainingConfig.dsm_weighting. "none" is the pre-Phase-2
                # unweighted objective, retained only for archive reproduction; it is
                # dominated by the smallest sigmas and cannot train the corrected
                # [0.01, 42] range.
                if config.dsm_weighting == "sigma_sq":
                    loss = torch.mean(((score - target) * sigma) ** 2)
                elif config.dsm_weighting == "none":
                    loss = torch.mean((score - target) ** 2)
                else:
                    raise ValueError(
                        f"Unknown dsm_weighting {config.dsm_weighting!r}; "
                        "expected 'sigma_sq' or 'none'."
                    )

            # NaN guard: skip optimizer step on non-finite loss.
            if not torch.isfinite(loss):
                nan_streak += 1
                logger.warning(
                    "NaN/Inf loss at epoch=%d step=%d (streak=%d); skipping batch.",
                    epoch, global_step, nan_streak,
                )
                if nan_streak >= config.nan_tolerance:
                    raise RuntimeError(
                        f"Training collapsed: {nan_streak} consecutive NaN/Inf losses at "
                        f"epoch={epoch} step={global_step}. "
                        f"Try lowering lr ({config.lr}) or sigma_max ({config.sigma_max})."
                    )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue
            nan_streak = 0

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                        config.grad_clip if config.grad_clip > 0 else float("inf")).item()
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
                    )
                else:
                    grad_norm = float(
                        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    )
                optimizer.step()
            if grad_norm > 10.0:
                logger.warning("High grad norm: %.2f at epoch=%d step=%d", grad_norm, epoch, global_step)
            if scheduler is not None:
                scheduler.step()

            if ema is not None:
                ema.update(model)

            loss_metric.update(loss.detach())
            global_step += 1
            if config.log_interval and global_step % config.log_interval == 0:
                avg = float(loss_metric.compute().item())
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info("epoch=%d step=%d loss=%.4f grad_norm=%.4f lr=%.2e",
                            epoch, global_step, avg, grad_norm, current_lr)
                _write_jsonl({
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": loss.item(),
                    "avg_loss": avg,
                    "grad_norm": grad_norm,
                    "lr": current_lr,
                    "wall_clock_seconds": time.time() - train_start,
                })

        avg_loss = float(loss_metric.compute().item())
        epoch_losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("epoch=%d avg_loss=%.4f lr=%.2e", epoch, avg_loss, current_lr)

        # Step 7B: validation DSM loss every N epochs.
        val_loss: Optional[float] = None
        if (config.val_every_n_epochs > 0 and epoch % config.val_every_n_epochs == 0
                and val_loader is not None):
            model.eval()
            val_metric = MeanMetric().to(device)
            with torch.no_grad():
                for val_batch in val_loader:
                    xv, _ = val_batch
                    xv = xv.to(device)
                    sv = _sample_sigma(xv, config.sigma_min, config.sigma_max)
                    nv = torch.randn_like(xv)
                    sv_out = model(xv + sv * nv, sv)
                    # Must use the SAME weighting as the training loss, or the
                    # overfit_ratio comparison below is between two different scales.
                    if config.dsm_weighting == "sigma_sq":
                        lv = torch.mean(((sv_out + nv / sv) * sv) ** 2)
                    else:
                        lv = torch.mean((sv_out + nv / sv) ** 2)
                    if torch.isfinite(lv):
                        val_metric.update(lv.detach())
            val_loss = float(val_metric.compute().item())
            overfit_ratio = val_loss / max(avg_loss, 1e-8)
            if overfit_ratio > 1.2:
                logger.warning(
                    "Possible overfitting: val_loss=%.4f vs train_loss=%.4f (ratio=%.2f)",
                    val_loss, avg_loss, overfit_ratio,
                )
            _write_jsonl({
                "epoch": epoch,
                "global_step": global_step,
                "val_dsm_loss": val_loss,
                "train_dsm_loss": avg_loss,
                "overfit_ratio": overfit_ratio,
                "lr": current_lr,
            })

        # Step 7C: sample diversity metric (optional).
        diversity: Optional[float] = None
        if config.log_sample_diversity:
            model.eval()
            with torch.no_grad():
                noise = torch.randn(16, *model.config.in_channels_shape_hint if hasattr(model, 'config') else (1, 28, 28), device=device)
                # Use a fixed sigma for diversity check
                sd = torch.full((16, 1, 1, 1), 0.5, device=device)
                samples = model(noise, sd).cpu()
                flat = samples.view(16, -1)
                # Mean pairwise L2 distance normalized by dimension
                diffs = flat.unsqueeze(0) - flat.unsqueeze(1)
                pairwise = diffs.norm(dim=2)
                diversity = float(pairwise.mean().item()) / flat.shape[1] ** 0.5
                if diversity < 0.1:
                    logger.warning("Low sample diversity=%.4f at epoch=%d — possible mode collapse.", diversity, epoch)
                _write_jsonl({"epoch": epoch, "global_step": global_step, "sample_diversity": diversity})

        if checkpoint_dir is not None:
            should_save = config.save_interval > 0 and epoch % config.save_interval == 0
            if should_save or epoch == config.epochs:
                ckpt_name = f"score_epoch_{epoch:04d}.pt"
                _save_checkpoint(checkpoint_dir / ckpt_name, epoch, model, optimizer, ema, global_step)

            # Wall-clock limit: save latest checkpoint and exit with code 42 so the
            # segmented launcher can restart from this point.
            if config.max_wall_hours > 0 and (time.time() - train_start) > config.max_wall_hours * 3600:
                _save_checkpoint(checkpoint_dir / "score_last.pt", epoch, model, optimizer, ema, global_step)
                logger.info(
                    "Wall-clock limit %.2f h reached at epoch=%d; saving score_last.pt and exiting 42.",
                    config.max_wall_hours, epoch,
                )
                sys.exit(42)

    if ema is not None:
        ema.copy_to(model)

    # Save score_best.pt (EMA weights baked in) as a stable pointer for eval scripts.
    if checkpoint_dir is not None:
        _save_checkpoint(checkpoint_dir / "score_best.pt", config.epochs, model, optimizer, ema, global_step)
        logger.info("Saved score_best.pt to %s", checkpoint_dir)

    final_stats = {
        "final_avg_loss": epoch_losses[-1],
        "best_avg_loss": min(epoch_losses),
        "epochs": len(epoch_losses),
        "device": resolved_device,
    }
    return final_stats
