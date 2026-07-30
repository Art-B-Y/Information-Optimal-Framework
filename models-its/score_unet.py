from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(max(1, groups), channels)


# ---------------------------------------------------------------------------
# Optional time embedding (Step 4A)
# ---------------------------------------------------------------------------

class _SinusoidalTimeEmbedding(nn.Module):
    """Normalized log-sigma sinusoidal time embedding with MLP projection.

    Maps a scalar sigma value to a D-dimensional embedding that is injected
    into each residual block via a learned affine (FiLM-style) modulation.
    Activated only when ``use_time_embedding=True`` in ``ScoreUNetConfig``.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.proj = nn.Sequential(
            nn.Linear(half * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: (B, 1, 1, 1) or broadcastable sigma tensor.

        Returns:
            Time embedding (B, embed_dim).
        """
        # Normalise to a scalar per-sample log value in a stable range.
        s = sigma.view(sigma.shape[0])  # (B,)
        log_s = torch.log(s.clamp(min=1e-5))  # (B,) log-sigma
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=s.device, dtype=s.dtype) / max(half - 1, 1)
        )
        args = log_s.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, embed_dim)
        return self.proj(emb)  # (B, embed_dim)


class _TimeModBlock(nn.Module):
    """Channel-wise affine modulation from a time embedding (FiLM)."""

    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(embed_dim, channels)
        self.shift = nn.Linear(embed_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), t_emb: (B, D)
        scale = self.scale(t_emb).view(t_emb.shape[0], -1, 1, 1) + 1.0
        shift = self.shift(t_emb).view(t_emb.shape[0], -1, 1, 1)
        return x * scale + shift


# ---------------------------------------------------------------------------
# Optional bottleneck self-attention (Step 4B)
# ---------------------------------------------------------------------------

class _BottleneckSelfAttention(nn.Module):
    """Multi-head self-attention applied at the U-Net bottleneck.

    Flattens spatial dimensions, runs ``nn.MultiheadAttention`` (4 heads),
    then reshapes back.  Activated only when ``use_attention=True``.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = _group_norm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        # Flatten spatial: (B, H*W, C)
        tokens = h.view(B, C, H * W).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        # Reshape back and add residual.
        return x + out.transpose(1, 2).view(B, C, H, W)


# ---------------------------------------------------------------------------
# Core U-Net blocks (unchanged from original)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _group_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScoreUNetConfig:
    in_channels: int = 3
    base_channels: int = 64
    channel_mults: Sequence[int] = (1, 2, 2, 4)
    dropout: float = 0.0
    # Step 4A: sinusoidal log-sigma time embedding injected per residual block.
    # Defaults to False for backward compatibility with existing checkpoints.
    use_time_embedding: bool = False
    time_embed_dim: int = 128
    # Step 4B: bottleneck self-attention.
    # Defaults to False for backward compatibility with existing checkpoints.
    use_attention: bool = False
    attention_heads: int = 4
    # Phase 2 (2026-07-15): preconditioning.
    #
    # "legacy" reproduces the pre-Phase-2 behaviour  h = x/sigma ; score = F(h)/sigma.
    # That is badly conditioned: with the DSM target -eps/sigma on x = x0 + sigma*eps,
    # the network input x/sigma = x0/sigma + eps is dominated by x0/sigma at small
    # sigma.  Measured on FashionMNIST (sigma_data ~= 0.70) the input magnitude spans
    # 70x at sigma=0.01 down to 1x at sigma=42 -- the network must extract eps from a
    # 70x-magnitude input at the low-sigma end, which is where sample quality is
    # decided.  This is a plausible contributor to the poor pre-audit FIDs.
    #
    # "edm" applies Karras et al. (2022) preconditioning, which holds the network
    # input at unit scale for EVERY sigma:
    #     c_in    = 1/sqrt(sigma^2 + sigma_data^2)
    #     c_skip  = sigma_data^2/(sigma^2 + sigma_data^2)
    #     c_out   = sigma*sigma_data/sqrt(sigma^2 + sigma_data^2)
    #     D(x;s)  = c_skip*x + c_out*F(c_in*x, c_noise)      (denoiser)
    #     score   = (D(x;s) - x)/sigma^2                     (Tweedie)
    # The returned quantity is still the score, so every downstream consumer
    # (DSM target -eps/sigma, the VE sampler, score_to_eps) is unchanged.
    preconditioning: str = "legacy"
    # Data standard deviation, used only by "edm" preconditioning.
    # FashionMNIST normalised to [-1,1] measures 0.70.
    sigma_data: float = 0.70


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ScoreUNet(nn.Module):
    def __init__(self, config: ScoreUNetConfig) -> None:
        super().__init__()
        self.config = config
        channels = [config.base_channels * mult for mult in config.channel_mults]

        # Optional time embedding (Step 4A).
        self.time_embed: Optional[_SinusoidalTimeEmbedding] = (
            _SinusoidalTimeEmbedding(config.time_embed_dim)
            if config.use_time_embedding
            else None
        )

        # Down path
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        # Time-modulation layers for down blocks (empty if time embedding disabled)
        self.down_time_mods = nn.ModuleList()
        in_ch = config.in_channels
        for idx, out_ch in enumerate(channels):
            self.down_blocks.append(DoubleConv(in_ch, out_ch, config.dropout))
            if config.use_time_embedding:
                self.down_time_mods.append(_TimeModBlock(out_ch, config.time_embed_dim))
            if idx < len(channels) - 1:
                self.downsamplers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            else:
                self.downsamplers.append(nn.Identity())
            in_ch = out_ch

        # Optional bottleneck attention (Step 4B).
        self.bottleneck_attn: Optional[_BottleneckSelfAttention] = (
            _BottleneckSelfAttention(channels[-1], config.attention_heads)
            if config.use_attention
            else None
        )

        # Up path
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.up_time_mods = nn.ModuleList()
        for idx in reversed(range(len(channels) - 1)):
            in_ch = channels[idx + 1]
            out_ch = channels[idx]
            self.upsamplers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(out_ch * 2, out_ch, config.dropout))
            if config.use_time_embedding:
                self.up_time_mods.append(_TimeModBlock(out_ch, config.time_embed_dim))

        self.output = nn.Conv2d(channels[0], config.in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma = torch.tensor([sigma], device=x.device, dtype=x.dtype)
        sigma = sigma.view(-1, *([1] * (x.dim() - 1)))
        # Clamp sigma away from zero to avoid NaN when sigma → 0 at the end of the
        # diffusion trajectory (audit fix CAT3-1 / CAT2-1).
        sigma = sigma.clamp(min=1e-5)

        edm = self.config.preconditioning == "edm"
        if edm:
            sd = self.config.sigma_data
            c_in = 1.0 / torch.sqrt(sigma ** 2 + sd ** 2)
            c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
            c_out = sigma * sd / torch.sqrt(sigma ** 2 + sd ** 2)
            # c_noise conditions the embedding on log-sigma (well-scaled across the
            # 4-decade sigma range) rather than on raw sigma.
            sigma_embed = torch.exp(0.25 * torch.log(sigma))
        else:
            sigma_embed = sigma

        # Compute time embedding once for all blocks (Step 4A).
        t_emb: Optional[torch.Tensor] = (
            self.time_embed(sigma_embed) if self.time_embed is not None else None
        )

        h = (c_in * x) if edm else (x / sigma)
        skips: list[torch.Tensor] = []
        for i, (block, down) in enumerate(zip(self.down_blocks, self.downsamplers)):
            h = block(h)
            if t_emb is not None:
                h = self.down_time_mods[i](h, t_emb)
            skips.append(h)
            h = down(h)

        h = skips.pop()  # bottleneck

        # Optional bottleneck self-attention (Step 4B).
        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)

        for i, (upsample, block) in enumerate(zip(self.upsamplers, self.up_blocks)):
            h = upsample(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)
            if t_emb is not None:
                h = self.up_time_mods[i](h, t_emb)

        out = self.output(h)
        if edm:
            # EDM denoiser -> score via Tweedie:  score = (D(x;s) - x)/sigma^2
            # with D = c_skip*x + c_out*F.  Computing it in that literal form suffers
            # CATASTROPHIC CANCELLATION at small sigma: c_skip -> 1, so (c_skip*x - x)
            # is a difference of near-equal floats, and dividing by sigma^2 amplifies
            # the rounding error (measured: ~3e-3 absolute at sigma=0.01 in float32).
            # Since c_skip - 1 = -sigma^2/(sigma^2 + sd^2) EXACTLY, the sigma^2
            # cancels analytically:
            #     score = -x/(sigma^2 + sd^2) + (c_out/sigma^2) * F
            # which is algebraically identical and numerically stable everywhere.
            sd = self.config.sigma_data
            return -x / (sigma ** 2 + sd ** 2) + (c_out / (sigma ** 2)) * out
        return out / sigma


def build_score_model(config: ScoreUNetConfig) -> ScoreUNet:
    return ScoreUNet(config)
