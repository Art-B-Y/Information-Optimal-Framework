from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from its.controllers import ControlConfig, build_control_policy
from its.controllers.neural_control import ConvControlConfig, ConvControlPolicy, build_conv_control_policy
from its.models import ScoreUNetConfig, build_score_model
from its.sde import ScoreSDEConfig
from its.sde.girsanov import girsanov_log_rn_step, girsanov_path_kl
from its.training.score_training import ExponentialMovingAverage, _sample_sigma
from its.utils import configure_logging
from its.utils.random import seed_everything
from its.eval import EvaluationConfig, evaluate_sampler
from its.eval.evaluator import evaluate_ddpm_baseline
from its.utils.wandb_logger import WandbLogger
from its.physics.fluctuation import jarzynski_work_estimate
from its.objectives.loss_components import (
    compute_trajectory_quality_loss,
    compute_reinforce_quality_loss,
    compute_v2_total_loss,
)
from its.training.schedules import WarmupSchedule, TwoPhaseScheduler


@dataclass
class ControlledScoreTrainingConfig:
    epochs: int = 5
    lr: float = 2e-4
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 100
    dataset_name: str = "fashionmnist"
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    # If > 0, use only this many samples from the training set.
    # Useful for fast sweep pilots and smoke tests.
    dataset_subset_size: int = 0
    model: ScoreUNetConfig = field(default_factory=lambda: ScoreUNetConfig(in_channels=1, base_channels=32))
    control: ControlConfig = field(default_factory=lambda: ControlConfig(state_dim=1, hidden_dim=64, depth=3, time_embedding_dim=8))
    sde: ScoreSDEConfig = field(default_factory=lambda: ScoreSDEConfig(beta_min=0.1, beta_max=5.0, num_steps=50, clamp=5.0, control_weight=1.0))
    control_weight: float = 1e-2
    quality_weight: float = 0.01
    # path_kl_weight > 0 is required for the information-theoretic objective to
    # contribute to training.  Setting it to 0.0 decouples the Girsanov path-KL
    # term from the loss (it is still logged), making it impossible for the
    # controller to learn from the path-distribution penalty regardless of how
    # long training runs.  Default raised from 0.0 to 0.1 (Bug 5 fix).
    path_kl_weight: float = 0.1
    # EMA of score model weights (Improvement 5).  Set to None to disable.
    ema_decay: Optional[float] = 0.999
    seed: int = 0
    mixed_precision: bool = True
    grad_accum_steps: int = 1
    checkpoint_dir: Optional[str] = None
    save_interval: int = 0
    resume_from: Optional[str] = None
    resume_latest: bool = False
    log_dir: Optional[str] = None
    run_name: Optional[str] = None
    log_file: Optional[str] = None
    eval_every: int = 0
    eval_num_samples: int = 256
    eval_batch_size: int = 64
    eval_baselines: list[str] = field(default_factory=list)
    wandb: bool = False
    wandb_project: str = "its"
    wandb_run_name: Optional[str] = None
    # JSONL step log — written every log_interval steps alongside the text log.
    jsonl_log: Optional[str] = None
    # Checkpoint every N epochs (None = use save_interval).
    save_every_n_epochs: Optional[int] = None
    # Max consecutive NaN batches before raising RuntimeError.
    nan_tolerance: int = 5
    # If set, load this pretrained score-model checkpoint before training.
    # Supports combined (score+control) or bare score-model checkpoints.
    score_backbone_ckpt: Optional[str] = None
    # If True, freeze the score model (requires_grad=False) and exclude its
    # parameters from the optimizer.  Useful when fine-tuning only the control
    # policy on top of a pretrained score backbone (Step 3B).
    freeze_score_model: bool = False
    # Early stopping based on path KL plateau (Step 2B, Session 5).
    # If path_kl has not decreased by more than early_stop_min_delta (relative)
    # over the last early_stop_patience epochs, training is stopped early.
    # Set early_stop_patience=0 to disable (default).
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.01  # 1% relative improvement threshold
    # Step 7A (Session 6): cosine annealing LR schedule over total steps.
    use_lr_schedule: bool = True
    lr_min: float = 1e-6
    # Step 7C (Session 6): log sample diversity per epoch.
    log_sample_diversity: bool = False
    # Ablation 5: use ConvControlPolicy instead of flat MLP.
    use_conv_control: bool = False
    conv_control: Optional[ConvControlConfig] = None

    # Session 10 v2 objective redesign.
    # "v1" = legacy DSM+CE+PathKL+quality; "v2" = DSM+traj_quality+reinforce+CE.detach()
    objective_version: str = "v1"
    detach_control_energy: bool = False
    reinforce_weight: float = 0.0
    trajectory_quality_temperature: float = 1.0
    # WarmupSchedule: phase_kl and reinforce weights ramp from 0 over warmup+ramp epochs.
    warmup_epochs: int = 0
    warmup_ramp_epochs: int = 0
    target_path_kl_weight: float = 0.01
    target_reinforce_weight: float = 0.1
    # TwoPhaseScheduler: low LR phase1, then cosine anneal from phase2_lr.
    use_two_phase_scheduler: bool = False
    phase1_epochs: int = 5
    phase1_lr: float = 1e-5
    phase2_lr: float = 5e-4
    # Scaled random init: output layer from N(0, init_std) instead of zeros.
    scaled_output_init: bool = False
    output_init_std: float = 1e-4
    # Joint fine-tuning: distillation regularizer keeps score near reference.
    distillation_weight: float = 0.0
    # LR multiplier for score model parameters when not frozen (joint finetune).
    score_lr_multiplier: float = 0.01

    # Speed overhaul (Session Speed).
    # torch.compile: adds ~1-2 min warmup, use for runs > 1 hour.
    use_compile: bool = False
    # TF32: Ampere+ GPUs only; ~10-20% speedup, safe for diffusion models.
    allow_tf32: bool = True

    # Fault tolerance: autosave every N minutes (0 = disable), wall-clock limit.
    checkpoint_every_n_minutes: int = 30
    # max_wall_hours > 0: save checkpoint and exit with code 42 at the limit.
    # Used by launch_segmented_training.py to run training in safe segments.
    max_wall_hours: float = 0.0


def _make_dataloader(cfg: ControlledScoreTrainingConfig):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    if cfg.dataset_name.lower() == "fashionmnist":
        mean, std = (0.5,), (0.5,)
        ds_train = datasets.FashionMNIST(
            root="./data/tensors",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
        )
    elif cfg.dataset_name.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_train = datasets.CIFAR10(
            root="./data/tensors",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
        )
    elif cfg.dataset_name.lower() == "mnist":
        mean, std = (0.1307,), (0.3081,)
        ds_train = datasets.MNIST(
            root="./data/tensors",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
        )
    else:
        raise ValueError(f"Unsupported dataset for controlled training: {cfg.dataset_name}")

    if cfg.dataset_subset_size > 0:
        from torch.utils.data import Subset
        import random as _random
        indices = _random.sample(range(len(ds_train)), min(cfg.dataset_subset_size, len(ds_train)))
        ds_train = Subset(ds_train, indices)

    _nw = cfg.num_workers
    _loader_kwargs: dict = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": _nw,
        "pin_memory": True,
    }
    if _nw > 0:
        _loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        _loader_kwargs["persistent_workers"] = cfg.persistent_workers
    loader = DataLoader(ds_train, **_loader_kwargs)
    return loader


def _resolve_log_file(config: ControlledScoreTrainingConfig) -> Optional[str]:
    if config.log_file:
        return config.log_file
    if not config.log_dir:
        return None
    name = config.run_name or "controlled_train"
    stamp = torch.randint(0, 1_000_000, (1,)).item()
    return str(Path(config.log_dir) / f"{name}_{stamp}.log")


class _JsonlWriter:
    """Append JSON-line records to a log file with a persistent buffered handle.

    Keeps a single open file descriptor to avoid the open/close syscall overhead
    on every write. Explicit flush() every N steps; auto-flush on close.
    """

    def __init__(self, path: Optional[str]) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        self._fh = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8", buffering=8192)

    def write(self, record: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record) + "\n")

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def __del__(self) -> None:
        self.close()


class _AutosaveThread(threading.Thread):
    """Background daemon thread that signals when a checkpoint save is needed.

    The main training loop polls `is_save_needed()` at batch boundaries and
    calls `save_done()` after completing the atomic autosave. Using a signal
    rather than direct model access keeps the save off the critical path.
    """

    def __init__(self, interval_minutes: int) -> None:
        super().__init__(daemon=True, name="autosave-timer")
        self.interval = interval_minutes * 60.0
        self._save_needed = threading.Event()
        self._stop = threading.Event()
        self._last_save = time.time()

    def run(self) -> None:
        while not self._stop.wait(60.0):
            if time.time() - self._last_save >= self.interval:
                self._save_needed.set()

    def is_save_needed(self) -> bool:
        return self._save_needed.is_set()

    def save_done(self) -> None:
        self._last_save = time.time()
        self._save_needed.clear()

    def stop(self) -> None:
        self._stop.set()


def _atomic_save(src: Path, dst: Path) -> None:
    """Atomically rename src to dst using shutil.move (crash-safe)."""
    shutil.move(str(src), str(dst))


def _save_autosave(
    ckpt_dir: Path,
    epoch: int,
    global_step: int,
    model: nn.Module,
    control: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    ema,
    rng_states: dict,
) -> None:
    """Write autosave checkpoint atomically: .tmp file then rename."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_dir / "autosave.pt.tmp"
    dst_path = ckpt_dir / "autosave.pt"
    payload: dict = {
        "epoch": epoch,
        "global_step": global_step,
        "score_state_dict": model.state_dict(),
        "control_state_dict": control.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model": model.state_dict(),
        "control": control.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_python": rng_states.get("python"),
        "rng_numpy": rng_states.get("numpy"),
        "rng_torch": rng_states.get("torch"),
        "rng_cuda": rng_states.get("cuda"),
        "is_autosave": True,
    }
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    if hasattr(model, "config"):
        payload["score_config"] = dataclasses.asdict(model.config)
    if hasattr(control, "config"):
        payload["control_config"] = dataclasses.asdict(control.config)
    torch.save(payload, tmp_path)
    shutil.move(str(tmp_path), str(dst_path))


def _capture_rng_states() -> dict:
    import random as _py_random
    import numpy as _np
    states: dict = {
        "python": _py_random.getstate(),
        "numpy": _np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state()
    return states


def _restore_rng_states(states: dict) -> None:
    import random as _py_random
    import numpy as _np
    if "python" in states and states["python"] is not None:
        _py_random.setstate(states["python"])
    if "numpy" in states and states["numpy"] is not None:
        _np.random.set_state(states["numpy"])
    if "torch" in states and states["torch"] is not None:
        torch.set_rng_state(states["torch"])
    if "cuda" in states and states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(states["cuda"])


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    control: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    global_step: int,
    ema: Optional[ExponentialMovingAverage] = None,
) -> None:
    payload: dict[str, object] = {
        "epoch": epoch,
        "global_step": global_step,
        # Canonical key names (Improvement 3).
        "score_state_dict": model.state_dict(),
        "control_state_dict": control.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # Legacy aliases for backward-compat readers.
        "model": model.state_dict(),
        "control": control.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    # Save architecture configs so checkpoints can be loaded without knowing
    # the original construction parameters (Bug 1 / Improvement 3 fix).
    if hasattr(model, "config"):
        payload["score_config"] = dataclasses.asdict(model.config)
    if hasattr(control, "config"):
        payload["control_config"] = dataclasses.asdict(control.config)
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    control: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    ema: Optional["ExponentialMovingAverage"] = None,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # Support both new (score_state_dict/control_state_dict) and legacy (model/control) keys.
    model.load_state_dict(checkpoint.get("score_state_dict", checkpoint["model"]))
    control.load_state_dict(checkpoint.get("control_state_dict", checkpoint["control"]))
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    return start_epoch, global_step


def _resolve_resume_path(config: ControlledScoreTrainingConfig) -> Optional[Path]:
    if config.resume_from:
        path = Path(config.resume_from)
        return path if path.is_file() else None
    if not config.resume_latest or not config.checkpoint_dir:
        return None
    ckpt_dir = Path(config.checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("controlled_epoch_*.pt"))
    return candidates[-1] if candidates else None


def _build_feature_extractor(device: torch.device) -> Optional[nn.Module]:
    """Return a frozen Inception-v3 feature extractor for quality matching.

    Returns ``None`` if torchvision is unavailable or loading fails.  The
    quality-matching term is silently skipped when this returns None.
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights

        incep = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        # Truncate at the average-pooling layer to get 2048-d feature vectors.
        incep.fc = nn.Identity()  # type: ignore[assignment]
        incep.aux_logits = False
        incep.eval()
        for p in incep.parameters():
            p.requires_grad_(False)
        return incep.to(device)
    except Exception:
        return None


def _feature_matching_loss(
    gen_samples: torch.Tensor,
    real_batch: torch.Tensor,
    extractor: nn.Module,
) -> torch.Tensor:
    """L2 distance between batch activation statistics of generated vs. real samples.

    Operates on the mean activation vector over the batch — a cheap per-step
    proxy for FID that does not require accumulating thousands of samples.

    Args:
        gen_samples: Generated images (B, C, H, W) in the model's value range.
        real_batch: Real images from the same batch (B, C, H, W).
        extractor: Frozen feature extractor (e.g. truncated Inception).

    Returns:
        Scalar L2 loss between mean activation vectors.
    """
    # Inception expects 299×299 RGB; resize & replicate channels if needed.
    import torch.nn.functional as F_func

    def _prepare(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2:] != (299, 299):
            x = F_func.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return x

    with torch.no_grad():
        real_feats = extractor(_prepare(real_batch.detach())).mean(dim=0)  # (2048,)

    gen_feats = extractor(_prepare(gen_samples)).mean(dim=0)  # (2048,)
    return F_func.mse_loss(gen_feats, real_feats)


def _load_score_backbone(ckpt_path: str, model: nn.Module, device: torch.device) -> None:
    """Load a pretrained score-model state dict from *ckpt_path* into *model*.

    Supports combined checkpoints (with ``score_state_dict`` / ``model_state_dict``
    / ``model`` keys) as well as bare state-dict files.  Only loads the score
    model weights; ignores optimizer, EMA, and control weights in combined files.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Score backbone checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        sd = (
            state.get("score_state_dict")
            or state.get("model_state_dict")
            or state.get("model")
        )
        if sd is None:
            sd = state  # assume the whole dict is the state dict
    else:
        sd = state
    model.load_state_dict(sd, strict=True)


def simulate_path(
    model: nn.Module,
    control_policy: nn.Module,
    config: ScoreSDEConfig,
    batch_size: int,
    device: torch.device,
    channels: int,
    height: int,
    width: int,
    return_log_probs: bool = False,
) -> tuple:
    """Run a differentiable controlled SDE simulation.

    Returns (final_x, control_energy, path_kl, jarzynski) by default.
    If return_log_probs=True, returns (final_x, control_energy, path_kl, jarzynski, log_rn)
    where log_rn is (B,) per-sample summed log Radon-Nikodym, used for REINFORCE.
    """
    step_size = 1.0 / config.num_steps
    ts = torch.linspace(1.0, config.eps, config.num_steps, device=device)
    # VE prior: x ~ N(0, sigma_max^2 I).  Phase 2 fix: this was N(0, I), which is only
    # the correct prior when sigma_max == 1.  With the corrected sigma_max=42 an
    # unscaled init would start the controller's rollout ~42x too close to the origin,
    # i.e. nowhere near the distribution the reverse trajectory expects at t=1.
    # This mirrors the same fix already applied to ScoreSDESimulator.sample.
    x = torch.randn(batch_size, channels, height, width, device=device) * config.sigma_max
    x.requires_grad_(True)
    control_energy_terms = []
    log_rn_terms = []
    for idx, t in enumerate(ts):
        t_batch = torch.full((batch_size, 1), t, device=device)
        noise = torch.randn_like(x)
        sigma = config.sigma_min * (config.sigma_max / config.sigma_min) ** t
        # Detach score: backbone is treated as fixed, gradient only flows through control policy.
        with torch.no_grad():
            score = model(x.detach(), sigma)

        # Audit 2026-07-15 (A1/A2): this previously used
        #   drift = 0.5*beta_t*(-x - score)   with a POSITIVE step while t descends,
        # which is not a reverse-time sampler at all -- with an exact analytic
        # score it converges to ~84x the true variance and never improves with
        # NFE.  The controller was therefore trained inside dynamics that never
        # generated data.  We now use the same correct VE reverse-diffusion step
        # as its.sde.score_sde, matching the VE kernel the score model was
        # trained on (score_training.py: noisy = x + sigma*noise).
        sigma_next = (
            config.sigma_min * (config.sigma_max / config.sigma_min) ** ts[idx + 1]
            if idx + 1 < len(ts)
            else torch.zeros_like(sigma)
        )
        dvar = (sigma ** 2 - sigma_next ** 2).clamp(min=0.0)
        drift = dvar * score.detach()

        # Support both MLP (flattened) and ConvControlPolicy (spatial) control policies.
        if isinstance(control_policy, ConvControlPolicy):
            control_4d = control_policy(x, t_batch)
            control_vec = control_4d.view(batch_size, -1)
        else:
            flat_x = x.view(batch_size, -1)
            control_vec = control_policy(flat_x, t_batch)
        control = control_vec.view_as(x)
        control_energy_terms.append((control.pow(2).mean()) * step_size)

        # The drift shift ACTUALLY integrated into x below.  Audit C6: the old
        # code passed the bare control_vec to the Girsanov term while the state
        # update used config.control_weight * control, making the exponent
        # inconsistent with the simulated SDE whenever control_weight != 1.
        control_term = config.control_weight * control * torch.sqrt(dvar)

        # Girsanov log Radon-Nikodym per sample (subsampled if requested).
        if config.jarzynski_subsample <= 1 or idx % config.jarzynski_subsample == 0:
            log_rn_terms.append(girsanov_log_rn_step(control_term, noise, dvar))

        x = x + drift + control_term + torch.sqrt(dvar) * noise

    control_energy = torch.stack(control_energy_terms).sum()
    log_rn = torch.stack(log_rn_terms, dim=0).sum(dim=0)  # (B,) per-sample log RN
    if config.jarzynski_subsample > 1:
        log_rn = log_rn * float(config.jarzynski_subsample)  # audit C6: rescale subsampled sum
    # Sign matched to girsanov_log_rn_step: KL = +mean(log_rn).  See sde/girsanov.py.
    path_kl = girsanov_path_kl(log_rn)
    jarzynski = jarzynski_work_estimate(log_rn.detach().cpu().numpy(), beta=1.0, clip=config.jarzynski_clip)
    if return_log_probs:
        return x, control_energy, path_kl, torch.tensor(jarzynski, device=device), log_rn
    return x, control_energy, path_kl, torch.tensor(jarzynski, device=device)


def train_controlled_score(config: ControlledScoreTrainingConfig) -> dict[str, float]:
    """Train score model + control policy jointly with DSM, control-energy, path-KL, and quality objectives."""
    log_file = _resolve_log_file(config)
    configure_logging(log_file=log_file)
    seed_everything(config.seed)
    resolved_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        resolved_device = "cpu"
    device = torch.device(resolved_device)

    # TF32: Ampere+ GPUs — safe 10-20% speedup alongside AMP.
    if config.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loader = _make_dataloader(config)
    model = build_score_model(config.model).to(device)
    # Determine image shape from dataset so we can assert control policy state_dim
    # matches flattened image size (Bug 2 fix).
    _dataset_image_shapes = {
        "fashionmnist": (1, 28, 28),
        "mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
    }
    image_shape = _dataset_image_shapes.get(config.dataset_name.lower())
    if config.use_conv_control:
        _channels = image_shape[0] if image_shape else 1
        conv_cfg = config.conv_control or ConvControlConfig(in_channels=_channels)
        control_policy = build_conv_control_policy(conv_cfg, device=device)
    else:
        control_policy = build_control_policy(config.control, device=device, image_shape=image_shape)
    logger = logging.getLogger(__name__)

    # torch.compile: fuses kernels for ~20-40% throughput gain on long runs.
    # Adds 1-2 min first-batch compilation overhead — logged as a warning.
    if config.use_compile and hasattr(torch, "compile"):
        try:
            logger.info("Compiling models with torch.compile(mode='reduce-overhead')... (~1-2 min)")
            model = torch.compile(model, mode="reduce-overhead")
            control_policy = torch.compile(control_policy, mode="reduce-overhead")
            logger.info("torch.compile succeeded.")
        except Exception as _ce:
            logger.warning("torch.compile failed (%s); continuing with uncompiled models.", _ce)

    # Load pretrained score backbone if requested (Step 3A).
    if config.score_backbone_ckpt:
        _load_score_backbone(config.score_backbone_ckpt, model, device)
        logger.info("Loaded pretrained score backbone from %s", config.score_backbone_ckpt)

    # Freeze score model if requested (Step 3B).
    if config.freeze_score_model:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        logger.info("Score model frozen -- only control policy will be trained.")

    # Session 10: scaled random init for controller output layer (break symmetry).
    if config.scaled_output_init and hasattr(control_policy, "output_proj"):
        with torch.no_grad():
            for p in control_policy.output_proj.parameters():
                if p.ndim >= 2:
                    nn.init.normal_(p, mean=0.0, std=config.output_init_std)
                else:
                    nn.init.zeros_(p)
        logger.info("Applied scaled output init (std=%.1e) to output_proj.", config.output_init_std)

    # Session 10: frozen reference score model for distillation regularizer.
    _reference_model: Optional[nn.Module] = None
    if config.distillation_weight > 0 and not config.freeze_score_model:
        import copy
        _reference_model = copy.deepcopy(model)
        for p in _reference_model.parameters():
            p.requires_grad_(False)
        _reference_model.eval()
        logger.info("Created frozen reference model for distillation (weight=%.2f).",
                    config.distillation_weight)

    # EMA over score model weights — used for evaluation and sampling (Improvement 5).
    ema: Optional[ExponentialMovingAverage] = (
        ExponentialMovingAverage(model, config.ema_decay) if config.ema_decay else None
    )
    # Build optimizer: joint finetune uses separate LR for score model (score_lr_multiplier).
    if config.freeze_score_model:
        trainable_params = list(control_policy.parameters())
        optim = torch.optim.AdamW(trainable_params, lr=config.lr)
    elif config.score_lr_multiplier != 1.0 and config.objective_version == "v2":
        optim = torch.optim.AdamW([
            {"params": control_policy.parameters(), "lr": config.lr},
            {"params": model.parameters(), "lr": config.lr * config.score_lr_multiplier},
        ])
    else:
        trainable_params = list(model.parameters()) + list(control_policy.parameters())
        optim = torch.optim.AdamW(trainable_params, lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    wandb = WandbLogger(enabled=config.wandb, project=config.wandb_project, run_name=config.wandb_run_name)

    # Session 10: WarmupSchedule for path_kl and reinforce weights.
    _warmup_schedule: Optional[WarmupSchedule] = None
    if config.warmup_epochs > 0 or config.warmup_ramp_epochs > 0:
        _warmup_schedule = WarmupSchedule(
            warmup_epochs=config.warmup_epochs,
            ramp_epochs=config.warmup_ramp_epochs,
            target_path_kl_weight=config.target_path_kl_weight,
            target_reinforce_weight=config.target_reinforce_weight,
        )
        logger.info("WarmupSchedule: warmup=%d ramp=%d target_pkL=%.4f target_reinf=%.4f",
                    config.warmup_epochs, config.warmup_ramp_epochs,
                    config.target_path_kl_weight, config.target_reinforce_weight)

    # LR scheduler: TwoPhaseScheduler (Session 10) or cosine annealing (legacy).
    _n_train_batches = len(loader) if hasattr(loader, "__len__") else 1
    _total_ctrl_steps = config.epochs * max(1, _n_train_batches)
    _two_phase_scheduler: Optional[TwoPhaseScheduler] = None
    if config.use_two_phase_scheduler:
        _two_phase_scheduler = TwoPhaseScheduler(
            optim,
            phase1_epochs=config.phase1_epochs,
            phase1_lr=config.phase1_lr,
            phase2_lr=config.phase2_lr,
            total_epochs=config.epochs,
            lr_min=config.lr_min,
        )
        ctrl_scheduler = None
        logger.info("TwoPhaseScheduler: phase1=%d epochs at lr=%.1e, phase2 cosine from %.1e",
                    config.phase1_epochs, config.phase1_lr, config.phase2_lr)
    else:
        ctrl_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=max(1, _total_ctrl_steps), eta_min=config.lr_min
            )
            if config.use_lr_schedule else None
        )
    # Frozen Inception feature extractor for quality matching (Improvement 2).
    # For v2: needed when quality_weight > 0 (trajectory quality) or reinforce_weight > 0 (rewards).
    # For v1: needed when quality_weight > 0 (feature matching).
    _need_extractor = (
        config.quality_weight > 0 or
        (config.objective_version == "v2" and config.reinforce_weight > 0)
    )
    feature_extractor: Optional[nn.Module] = (
        _build_feature_extractor(device) if _need_extractor else None
    )

    start_epoch = 1
    global_step = 0
    if config.checkpoint_dir:
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    resume_path = _resolve_resume_path(config)
    if resume_path:
        start_epoch, global_step = _load_checkpoint(
            resume_path, model, control_policy, optim,
            scaler if scaler.is_enabled() else None, device, ema=ema,
        )

    accum_steps = max(1, int(config.grad_accum_steps))
    save_period = config.save_every_n_epochs or config.save_interval
    nan_streak = 0
    train_start = time.time()

    # Early stopping state (Step 2B, Session 5).
    # Tracks per-epoch mean path_kl to detect plateaus.
    _early_stop_path_kl_history: list[float] = []
    _early_stopped = False

    # Determine JSONL log path: explicit > log_dir / run_name_timestamp.jsonl
    jsonl_path = config.jsonl_log
    if jsonl_path is None and config.log_dir:
        name = config.run_name or "controlled_train"
        jsonl_path = str(Path(config.log_dir) / f"{name}_{int(time.time())}.jsonl")
    jsonl_writer = _JsonlWriter(jsonl_path)

    # Fixed diagnostic batch for per-epoch cosine-similarity computation.
    _diag_batch: Optional[torch.Tensor] = None

    # Autosave thread — signals every checkpoint_every_n_minutes.
    _ckpt_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    _autosave_thread: Optional[_AutosaveThread] = None
    if _ckpt_dir is not None and config.checkpoint_every_n_minutes > 0:
        _autosave_thread = _AutosaveThread(config.checkpoint_every_n_minutes)
        _autosave_thread.start()
        logger.info("Autosave thread started: every %d min -> %s/autosave.pt",
                    config.checkpoint_every_n_minutes, _ckpt_dir)

    def _save_ckpt_and_latest(path: Path) -> None:
        """Save checkpoint at *path* and update controlled_last.pt in same dir."""
        _save_checkpoint(
            path, epoch, model, control_policy, optim,
            scaler if scaler.is_enabled() else None, global_step, ema=ema,
        )
        latest = path.parent / "controlled_last.pt"
        shutil.copy2(str(path), str(latest))

    for epoch in range(start_epoch, config.epochs + 1):
        if not config.freeze_score_model:
            model.train()
        control_policy.train()
        optim.zero_grad(set_to_none=True)

        # Session 10: per-epoch LR and weight scheduling.
        if _two_phase_scheduler is not None:
            current_lr = _two_phase_scheduler.step(epoch - 1)  # 0-indexed
        # WarmupSchedule: compute effective path_kl and reinforce weights for this epoch.
        _epoch_pkL_weight = config.path_kl_weight
        _epoch_reinforce_weight = config.reinforce_weight
        if _warmup_schedule is not None:
            _scheduled = _warmup_schedule.get_weights(epoch - 1)  # 0-indexed
            _epoch_pkL_weight = _scheduled["path_kl_weight"]
            _epoch_reinforce_weight = _scheduled["reinforce_weight"]

        for batch_idx, batch in enumerate(loader, start=1):
            x, _ = batch
            x = x.to(device)

            # Cache first batch as fixed diagnostic batch.
            if _diag_batch is None:
                _diag_batch = x[:16].detach()

            sigma = _sample_sigma(x, config.sigma_min, config.sigma_max)
            noise = torch.randn_like(x)
            noisy = x + sigma * noise
            target = -noise / sigma

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                score = model(noisy, sigma)
                dsm_loss = torch.mean((score - target) ** 2)

                # Differentiable controlled sampling path.
                # v2: also return per-sample log_rn for REINFORCE.
                _return_lp = (config.objective_version == "v2" and _epoch_reinforce_weight > 0)
                _path_result = simulate_path(
                    model, control_policy, config.sde,
                    batch_size=x.shape[0], device=device,
                    channels=x.shape[1], height=x.shape[2], width=x.shape[3],
                    return_log_probs=_return_lp,
                )
                if _return_lp:
                    gen_x, control_energy, path_kl, jarzynski, _log_rn = _path_result
                else:
                    gen_x, control_energy, path_kl, jarzynski = _path_result
                    _log_rn = None

                if config.objective_version == "v2":
                    # v2: trajectory quality (per-sample nearest-neighbour feature distance).
                    trajectory_quality_loss = torch.tensor(0.0, device=device)
                    reinforce_quality_loss = torch.tensor(0.0, device=device)
                    if feature_extractor is not None:
                        try:
                            _need_per_sample = (_log_rn is not None and _epoch_reinforce_weight > 0)
                            _tq_result = compute_trajectory_quality_loss(
                                gen_x, x, feature_extractor,
                                temperature=config.trajectory_quality_temperature,
                                return_per_sample=_need_per_sample,
                            )
                            if _need_per_sample:
                                trajectory_quality_loss, _tq_per_sample = _tq_result
                                # REINFORCE: reward = negative per-sample quality distance.
                                _rewards = -_tq_per_sample  # (B,) higher = closer to real
                                reinforce_quality_loss = compute_reinforce_quality_loss(
                                    _rewards, _log_rn,
                                )
                            else:
                                trajectory_quality_loss = _tq_result
                        except Exception as _exc:
                            logger.debug("v2 quality loss failed: %s", _exc)

                    # v2 total loss: control_energy is detached if requested.
                    loss, _loss_components = compute_v2_total_loss(
                        dsm_loss=dsm_loss,
                        path_kl=path_kl,
                        trajectory_quality_loss=trajectory_quality_loss,
                        reinforce_quality_loss=reinforce_quality_loss,
                        control_energy=control_energy,
                        path_kl_weight=_epoch_pkL_weight,
                        quality_weight=config.quality_weight,
                        reinforce_weight=_epoch_reinforce_weight,
                        detach_control_energy=config.detach_control_energy,
                    )
                    loss = loss / accum_steps
                    # For logging: alias v2 fields to legacy names.
                    quality_loss = trajectory_quality_loss
                else:
                    # v1 legacy objective.
                    quality_loss = torch.tensor(0.0, device=device)
                    if feature_extractor is not None and config.quality_weight > 0:
                        try:
                            quality_loss = _feature_matching_loss(gen_x, x, feature_extractor)
                        except Exception:
                            pass

                    _ce = control_energy.detach() if config.detach_control_energy else control_energy
                    loss = (
                        dsm_loss
                        + config.control_weight * _ce
                        + _epoch_pkL_weight * path_kl
                        + config.quality_weight * quality_loss
                    ) / accum_steps

                # Session 10 joint fine-tune: distillation regularizer.
                if _reference_model is not None and config.distillation_weight > 0:
                    with torch.no_grad():
                        _ref_score = _reference_model(noisy, sigma)
                    _distill = torch.nn.functional.mse_loss(score, _ref_score)
                    loss = loss + (config.distillation_weight * _distill) / accum_steps

            # NaN guard: skip optimizer step for bad batches (Step 5B).
            if not torch.isfinite(loss):
                nan_streak += 1
                logger.warning(
                    "NaN/Inf loss at epoch=%d step=%d (streak=%d); skipping batch.",
                    epoch, global_step, nan_streak,
                )
                if nan_streak >= config.nan_tolerance:
                    raise RuntimeError(
                        f"Training collapsed: {nan_streak} consecutive NaN batches. "
                        f"Hyperparams: control_weight={config.control_weight}, "
                        f"path_kl_weight={config.path_kl_weight}, "
                        f"quality_weight={config.quality_weight}, lr={config.lr}"
                    )
                optim.zero_grad(set_to_none=True)
                global_step += 1
                continue
            nan_streak = 0

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_norm_score = 0.0
            grad_norm_control = 0.0
            if batch_idx % accum_steps == 0:
                if config.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optim)
                    grad_norm_score = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
                    )
                    grad_norm_control = float(
                        torch.nn.utils.clip_grad_norm_(control_policy.parameters(), config.grad_clip).item()
                    )
                else:
                    grad_norm_score = float(
                        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    )
                    grad_norm_control = float(
                        sum(p.grad.norm().item() ** 2 for p in control_policy.parameters() if p.grad is not None) ** 0.5
                    )
                if grad_norm_score > 20.0:
                    logger.warning("High score grad norm: %.2f at step=%d", grad_norm_score, global_step)
                if grad_norm_control > 20.0:
                    logger.warning("High control grad norm: %.2f at step=%d", grad_norm_control, global_step)

                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                if ctrl_scheduler is not None:
                    ctrl_scheduler.step()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            global_step += 1
            # Keep detached tensor; only call .item() (CPU-GPU sync) at log boundary.
            _loss_for_log = loss.detach() * accum_steps

            if config.log_interval and global_step % config.log_interval == 0:
                total_loss_val = float(_loss_for_log.item())
                logger.info(
                    "epoch=%d step=%d loss=%.4f dsm=%.4f control=%.4f path_kl=%.4f jar=%.4f quality=%.4f",
                    epoch, global_step, total_loss_val,
                    dsm_loss.item(), control_energy.item(),
                    path_kl.item(), jarzynski.item(), quality_loss.item(),
                )
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "total_loss": total_loss_val,
                    "dsm_loss": dsm_loss.item(),
                    "control_energy": control_energy.item(),
                    "path_kl": path_kl.item(),
                    "quality_loss": quality_loss.item(),
                    "jarzynski": jarzynski.item(),
                    "grad_norm_score": grad_norm_score,
                    "grad_norm_control": grad_norm_control,
                    "lr": optim.param_groups[0]["lr"],
                    "wall_clock_seconds": time.time() - train_start,
                    "objective_version": config.objective_version,
                    "epoch_pkL_weight": _epoch_pkL_weight,
                    "epoch_reinforce_weight": _epoch_reinforce_weight,
                }
                jsonl_writer.write(record)
                jsonl_writer.flush()
                wandb.log({**record}, step=global_step)

            # Autosave: atomic write every checkpoint_every_n_minutes.
            if _autosave_thread is not None and _autosave_thread.is_save_needed():
                _save_autosave(
                    _ckpt_dir, epoch, global_step, model, control_policy,
                    optim, scaler if scaler.is_enabled() else None, ema,
                    _capture_rng_states(),
                )
                _autosave_thread.save_done()
                logger.info("Autosave written at epoch=%d step=%d.", epoch, global_step)

            # Wall-clock limit: exit cleanly with code 42 for segmented launcher.
            if config.max_wall_hours > 0 and (time.time() - train_start) > config.max_wall_hours * 3600:
                logger.info(
                    "Wall-clock limit %.2f h reached at epoch=%d step=%d. "
                    "Saving segment-end checkpoint and exiting with code 42.",
                    config.max_wall_hours, epoch, global_step,
                )
                if _ckpt_dir is not None:
                    _save_ckpt_and_latest(_ckpt_dir / f"controlled_epoch_{epoch:04d}.pt")
                jsonl_writer.close()
                if _autosave_thread is not None:
                    _autosave_thread.stop()
                sys.exit(42)

            if config.eval_every and global_step % config.eval_every == 0:
                eval_cfg = EvaluationConfig(
                    dataset_name=config.dataset_name,
                    num_samples=config.eval_num_samples,
                    batch_size=config.eval_batch_size,
                    device=resolved_device,
                )
                if ema is not None:
                    ema.copy_to(model)
                eval_results = evaluate_sampler(model, control_policy, config.sde, eval_cfg)
                wandb.log({f"eval/{k}": v for k, v in eval_results.items()}, step=global_step)
                for baseline in config.eval_baselines:
                    if baseline in {"ddpm", "ddim"}:
                        ddpm_results = evaluate_ddpm_baseline(model, config.sde, eval_cfg, baseline=baseline)
                        wandb.log({f"eval_{baseline}/{k}": v for k, v in ddpm_results.items()}, step=global_step)

        logger.info("epoch=%d done", epoch)

        # Per-epoch cosine-similarity diagnostic.
        if _diag_batch is not None:
            try:
                model.eval()
                control_policy.eval()
                with torch.no_grad():
                    diag_x = _diag_batch.to(device)
                    diag_sigma = torch.full((diag_x.shape[0], 1, 1, 1), 0.5, device=device)
                    score_vecs = model(diag_x, diag_sigma)
                    if isinstance(control_policy, ConvControlPolicy):
                        ctrl_vecs = control_policy(diag_x, torch.zeros(diag_x.shape[0], 1, device=device))
                    else:
                        ctrl_vecs = control_policy(
                            diag_x.view(diag_x.shape[0], -1),
                            torch.zeros(diag_x.shape[0], 1, device=device),
                        ).view_as(diag_x)
                    s_flat = score_vecs.view(score_vecs.shape[0], -1)
                    c_flat = ctrl_vecs.view(ctrl_vecs.shape[0], -1)
                    cos_sim = torch.nn.functional.cosine_similarity(s_flat, c_flat, dim=1).mean().item()
                    score_mag = s_flat.norm(dim=1).mean().item()
                    ctrl_mag = c_flat.norm(dim=1).mean().item()
                    mag_ratio = ctrl_mag / max(score_mag, 1e-8)
                diag_record = {
                    "epoch": epoch, "global_step": global_step,
                    "cos_sim_score_ctrl": cos_sim,
                    "score_drift_mag": score_mag,
                    "ctrl_drift_mag": ctrl_mag,
                    "ctrl_score_mag_ratio": mag_ratio,
                    "wall_clock_seconds": time.time() - train_start,
                }
                logger.info(
                    "epoch=%d diag: cos_sim=%.4f ctrl_mag=%.4f score_mag=%.4f ratio=%.4f",
                    epoch, cos_sim, ctrl_mag, score_mag, mag_ratio,
                )
                jsonl_writer.write(diag_record)
                model.train()
                control_policy.train()
            except Exception as exc:
                logger.warning("Epoch diagnostic failed: %s", exc)

        if config.checkpoint_dir and save_period > 0 and epoch % save_period == 0:
            ckpt_path = Path(config.checkpoint_dir) / f"controlled_epoch_{epoch:04d}.pt"
            _save_ckpt_and_latest(ckpt_path)

        # Early stopping: check path KL plateau (Step 2B, Session 5).
        if config.early_stop_patience > 0:
            epoch_path_kl = float(path_kl.item())
            _early_stop_path_kl_history.append(epoch_path_kl)
            if len(_early_stop_path_kl_history) > config.early_stop_patience:
                window = _early_stop_path_kl_history[-config.early_stop_patience:]
                best_recent = min(window)
                oldest_recent = window[0]
                # Relative improvement from window start to window best.
                if oldest_recent > 0:
                    rel_improvement = (oldest_recent - best_recent) / oldest_recent
                else:
                    rel_improvement = 0.0
                if rel_improvement < config.early_stop_min_delta:
                    logger.warning(
                        "Early stopping at epoch %d: path_kl improved only %.4f%% "
                        "over last %d epochs (threshold: %.1f%%). "
                        "Information-theoretic objective has plateaued.",
                        epoch, rel_improvement * 100,
                        config.early_stop_patience, config.early_stop_min_delta * 100,
                    )
                    _early_stopped = True
                    break

    # Apply EMA weights before final save.
    if ema is not None:
        ema.copy_to(model)

    if config.checkpoint_dir:
        ckpt_path = Path(config.checkpoint_dir) / "controlled_last.pt"
        _save_checkpoint(
            ckpt_path, epoch, model, control_policy, optim,
            scaler if scaler.is_enabled() else None, global_step, ema=ema,
        )

    jsonl_writer.close()
    if _autosave_thread is not None:
        _autosave_thread.stop()

    return {
        "loss": float(loss.item() * accum_steps),
        "dsm_loss": float(dsm_loss.item()),
        "control_energy": float(control_energy.item()),
        "path_kl": float(path_kl.item()),
        "jarzynski": float(jarzynski.item()),
        "quality_loss": float(quality_loss.item()),
    }
