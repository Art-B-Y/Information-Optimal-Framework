from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


# ---------------------------------------------------------------------------
# Shared time-embedding modules
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)
        self.act = nn.Tanh()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(t))


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by a small MLP projection."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.proj = nn.Sequential(
            nn.Linear(half * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1) or (B,)
        if t.dim() > 1:
            t = t.squeeze(-1)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, embed_dim)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ControlConfig:
    """Configuration for the MLP-based flattened control policy (toy experiments / fallback)."""

    state_dim: int
    hidden_dim: int = 128
    depth: int = 3
    activation: str = "tanh"
    time_embedding_dim: int = 4
    dropout: float = 0.0


@dataclass
class ConvControlConfig:
    """Configuration for the convolutional control policy (image experiments).

    The network accepts (B, in_channels, H, W) image tensors directly and
    injects time information via Feature-wise Linear Modulation (FiLM).
    """

    in_channels: int = 3
    base_channels: int = 64
    num_blocks: int = 3
    time_embed_dim: int = 128
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# MLP policy (kept for toy experiments)
# ---------------------------------------------------------------------------

class ControlPolicy(nn.Module):
    """MLP control policy operating on flattened state vectors.

    Suitable for toy experiments.  For image data use ``ConvControlPolicy``
    which preserves spatial structure and is far more parameter-efficient.
    """

    def __init__(self, config: ControlConfig) -> None:
        super().__init__()
        self.config = config
        act_factory: Callable[[], nn.Module] = _ACTIVATIONS.get(config.activation.lower(), nn.Tanh)
        blocks: list[nn.Module] = []
        in_dim = config.state_dim + config.time_embedding_dim
        hidden = config.hidden_dim
        for layer_idx in range(max(config.depth - 1, 1)):
            blocks.append(nn.Linear(in_dim if layer_idx == 0 else hidden, hidden))
            blocks.append(act_factory())
            if config.dropout > 0:
                blocks.append(nn.Dropout(config.dropout))
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden if blocks else in_dim, config.state_dim)
        self.time_embed = TimeEmbedding(config.time_embedding_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_embed = self.time_embed(t)
        xt = torch.cat([x, t_embed], dim=-1)
        features = self.backbone(xt) if len(self.backbone) > 0 else xt
        return self.head(features)


# ---------------------------------------------------------------------------
# Convolutional policy (image experiments)
# ---------------------------------------------------------------------------

class _FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale + shift via time embedding."""

    def __init__(self, channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.scale_proj = nn.Linear(time_embed_dim, channels)
        self.shift_proj = nn.Linear(time_embed_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), t_emb: (B, D)
        scale = self.scale_proj(t_emb).view(t_emb.shape[0], -1, 1, 1) + 1.0
        shift = self.shift_proj(t_emb).view(t_emb.shape[0], -1, 1, 1)
        return x * scale + shift


class _ResBlock(nn.Module):
    """Residual conv block with GroupNorm, SiLU, and FiLM time conditioning."""

    def __init__(self, channels: int, time_embed_dim: int, dropout: float) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups //= 2
        self.norm1 = nn.GroupNorm(max(1, groups), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = _FiLMLayer(channels, time_embed_dim)
        self.norm2 = nn.GroupNorm(max(1, groups), channels)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, t_emb)
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return x + h


class ConvControlPolicy(nn.Module):
    """Lightweight convolutional control policy for image-space SDEs.

    Architecture:
    - Input projection: 1×1 conv from ``in_channels`` to ``base_channels``
    - ``num_blocks`` residual conv blocks, each FiLM-conditioned on time
    - Output projection: 1×1 conv back to ``in_channels``

    Accepts (B, C, H, W) tensors directly — no flattening required.
    """

    def __init__(self, config: ConvControlConfig) -> None:
        super().__init__()
        self.config = config
        C = config.base_channels

        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        self.input_proj = nn.Conv2d(config.in_channels, C, kernel_size=1)
        self.blocks = nn.ModuleList([
            _ResBlock(C, config.time_embed_dim, config.dropout)
            for _ in range(config.num_blocks)
        ])
        self.output_proj = nn.Conv2d(C, config.in_channels, kernel_size=1)
        # Zero-initialise output so the controller starts as a near-zero perturbation.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, C, H, W).
            t: Time tensor (B, 1) or (B,) in [0, 1].

        Returns:
            Control signal (B, C, H, W).
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_control_policy(
    config: ControlConfig,
    device: Optional[torch.device] = None,
    image_shape: Optional[Sequence[int]] = None,
) -> ControlPolicy:
    """Build an MLP ``ControlPolicy`` from *config*.

    Args:
        config: Policy configuration (must have ``state_dim`` set correctly).
        device: Optional target device.
        image_shape: If provided, assert ``config.state_dim == prod(image_shape)``
            so that MLP policies flattening image tensors are correctly sized.
            E.g. pass ``(3, 32, 32)`` for CIFAR-10 to catch mismatches early.
    """
    if image_shape is not None:
        expected_dim = math.prod(image_shape)
        assert config.state_dim == expected_dim, (
            f"ControlConfig.state_dim={config.state_dim} does not match "
            f"math.prod(image_shape={image_shape})={expected_dim}. "
            "Set state_dim to the flattened image size (784 for FashionMNIST, 3072 for CIFAR-10)."
        )
    policy = ControlPolicy(config)
    if device is not None:
        policy.to(device)
    return policy


def build_conv_control_policy(
    config: ConvControlConfig,
    device: Optional[torch.device] = None,
) -> ConvControlPolicy:
    """Build a ``ConvControlPolicy`` from *config*.

    Preferred over ``build_control_policy`` for image-space experiments because
    it preserves spatial structure and avoids the memory/compute overhead of
    flattening high-resolution images to 1-D vectors.
    """
    policy = ConvControlPolicy(config)
    if device is not None:
        policy.to(device)
    return policy
