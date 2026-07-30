"""Training schedules for Session 10 redesigned objective (Step 3).

Provides:
  WarmupSchedule  — linearly ramps path_kl_weight and reinforce_weight from 0
                    over the first warmup+ramp epochs to prevent early collapse.
  TwoPhaseScheduler — PyTorch LR scheduler wrapper that uses a low LR for the
                    first N epochs then jumps to a higher LR + cosine annealing.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.optim as optim


class WarmupSchedule:
    """Per-epoch weight scheduler with warm-up and linear ramp phases.

    During the first ``warmup_epochs`` epochs (0-indexed: epochs 0 to warmup-1):
      - ``path_kl_weight`` = 0.0
      - ``reinforce_weight`` = 0.0
      (Only DSM + trajectory_quality_loss contribute gradients.)

    During the next ``ramp_epochs`` epochs:
      - Both weights linearly ramp from 0 to their target values.

    After warmup + ramp epochs:
      - Both weights are at their target values permanently.

    Args:
        warmup_epochs: Number of epochs at zero weight (default 5).
        ramp_epochs: Number of epochs to linearly ramp to target (default 5).
        target_path_kl_weight: Final path_kl_weight after ramp (default 0.01).
        target_reinforce_weight: Final reinforce_weight after ramp (default 0.1).
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        ramp_epochs: int = 5,
        target_path_kl_weight: float = 0.01,
        target_reinforce_weight: float = 0.1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.target_path_kl_weight = target_path_kl_weight
        self.target_reinforce_weight = target_reinforce_weight

    def get_weights(self, epoch: int) -> dict[str, float]:
        """Return the scheduled loss weights for the given epoch (0-indexed).

        Args:
            epoch: Current epoch (0-indexed: first epoch is 0).

        Returns:
            Dict with keys 'path_kl_weight' and 'reinforce_weight'.
        """
        if epoch < self.warmup_epochs:
            return {"path_kl_weight": 0.0, "reinforce_weight": 0.0}

        epochs_into_ramp = epoch - self.warmup_epochs
        if epochs_into_ramp >= self.ramp_epochs:
            return {
                "path_kl_weight": self.target_path_kl_weight,
                "reinforce_weight": self.target_reinforce_weight,
            }

        # Linear ramp: [0, ramp_epochs) → [0, target]
        frac = (epochs_into_ramp + 1) / self.ramp_epochs
        return {
            "path_kl_weight": frac * self.target_path_kl_weight,
            "reinforce_weight": frac * self.target_reinforce_weight,
        }

    def __call__(self, epoch: int) -> dict[str, float]:
        return self.get_weights(epoch)


class TwoPhaseScheduler:
    """Two-phase learning rate scheduler.

    Phase 1 (epochs 0 to ``phase1_epochs - 1``):
      - Learning rate = ``phase1_lr`` (very small, allows direction finding).

    Phase 2 (epoch >= ``phase1_epochs``):
      - Learning rate jumps to ``phase2_lr`` at epoch ``phase1_epochs``.
      - Then applies cosine annealing from ``phase2_lr`` down to ``lr_min``
        over the remaining ``total_epochs - phase1_epochs`` epochs.

    Usage:
        scheduler = TwoPhaseScheduler(optimizer, phase1_epochs=5,
                                      phase1_lr=1e-5, phase2_lr=5e-4,
                                      total_epochs=100)
        for epoch in range(total_epochs):
            scheduler.step(epoch)
            # optimizer.param_groups[0]['lr'] is now set
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        phase1_epochs: int = 5,
        phase1_lr: float = 1e-5,
        phase2_lr: float = 5e-4,
        total_epochs: int = 100,
        lr_min: float = 5e-7,
    ) -> None:
        self.optimizer = optimizer
        self.phase1_epochs = phase1_epochs
        self.phase1_lr = phase1_lr
        self.phase2_lr = phase2_lr
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self._phase2_epochs = max(1, total_epochs - phase1_epochs)
        # Set initial LR
        self._set_lr(phase1_lr)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self, epoch: int) -> float:
        """Update optimizer learning rate for the given epoch (0-indexed).

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Current learning rate.
        """
        if epoch < self.phase1_epochs:
            lr = self.phase1_lr
        else:
            # Cosine annealing in phase 2
            t = epoch - self.phase1_epochs
            T = self._phase2_epochs
            lr = self.lr_min + 0.5 * (self.phase2_lr - self.lr_min) * (
                1 + math.cos(math.pi * t / T)
            )
        self._set_lr(lr)
        return lr

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
