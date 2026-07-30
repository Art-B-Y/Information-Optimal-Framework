from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from its.sde import ScoreSDESimulator, ScoreSDEConfig


try:  # torchmetrics optional dependency
    from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
    from torchmetrics.image.inception import InceptionScore  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "torchmetrics is required for evaluation. Install with `pip install torchmetrics`."
    ) from exc

logger = logging.getLogger(__name__)

# Minimum recommended sample count for reliable FID.  Below this threshold a
# warning is emitted (Step 6A).
_MIN_RELIABLE_SAMPLES = 2048

# Default sample counts per dataset (Improvement 4).
# These are set high enough for reliable FID estimates.  Use --quick (256) for
# fast smoke tests during development.
_DEFAULT_NUM_SAMPLES: dict[str, int] = {
    "fashionmnist": 5000,
    "mnist": 5000,
    "cifar10": 10000,
}

# Full-eval sample counts — used when --full-eval is passed (Step 6D).
_FULL_EVAL_NUM_SAMPLES: dict[str, int] = {
    "fashionmnist": 5000,
    "mnist": 5000,
    "cifar10": 10000,
}


@dataclass
class EvaluationConfig:
    dataset_name: str
    # Defaults to dataset-specific count (see _DEFAULT_NUM_SAMPLES) if -1.
    num_samples: int = -1
    batch_size: int = 64
    device: str = "cpu"
    # Optional RNG seed for reproducible FID/IS (Step 6C).
    seed: Optional[int] = None
    # When True, override num_samples to _FULL_EVAL_NUM_SAMPLES and compute
    # dual-seed FID variance (Step 6D).
    full_eval: bool = False

    def resolved_num_samples(self) -> int:
        if self.full_eval:
            return _FULL_EVAL_NUM_SAMPLES.get(self.dataset_name.lower(), 5000)
        if self.num_samples > 0:
            return self.num_samples
        return _DEFAULT_NUM_SAMPLES.get(self.dataset_name.lower(), 5000)


def _make_dataset(dataset_name: str, batch_size: int) -> tuple[torch.utils.data.DataLoader, int, int, int]:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    name = dataset_name.lower()
    # Always use the TEST split for real-image collection so FID is not
    # artificially inflated by memorised training images (audit fix CAT1-1 / CAT4-1).
    if name in {"fashionmnist", "mnist"}:
        channels, height, width = 1, 28, 28
        if name == "fashionmnist":
            dataset = datasets.FashionMNIST(
                root="./data/tensors",
                train=False,  # test split — not train
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
            )
        else:
            dataset = datasets.MNIST(
                root="./data/tensors",
                train=False,  # test split — not train
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            )
    elif name == "cifar10":
        channels, height, width = 3, 32, 32
        dataset = datasets.CIFAR10(
            root="./data/tensors",
            train=False,  # test split — not train
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset for evaluation: {dataset_name}")
    # Assertion guard: verify the test-split flag was respected (should never trip,
    # but catches future regressions if dataset builders change behaviour).
    assert not getattr(dataset, "train", True), (
        f"Real-image dataset for FID must use the test split (train=False). "
        f"Got train=True for dataset '{dataset_name}'."
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return loader, channels, height, width


def evaluate_sampler(
    score_model: torch.nn.Module,
    control_policy: Optional[torch.nn.Module],
    sde_config: ScoreSDEConfig,
    eval_cfg: EvaluationConfig,
) -> dict[str, float]:
    """Run controlled SDE sampling and compute FID, IS, NFE, and thermodynamic metrics.

    Step 6A: Emits a warning when num_samples < 2048 (unreliable FID territory).
    Step 6C: Respects eval_cfg.seed for reproducible sampling.
    Step 6D: When eval_cfg.full_eval=True, uses full-eval sample counts.
    Step 6E: When eval_cfg.full_eval=True, runs a second seed and reports FID
             variance as fid_std.
    """
    device = torch.device(eval_cfg.device)
    score_model.eval()
    if control_policy is not None:
        control_policy.eval()

    loader, channels, height, width = _make_dataset(eval_cfg.dataset_name, eval_cfg.batch_size)
    simulator = ScoreSDESimulator(score_model, sde_config)
    num_samples = eval_cfg.resolved_num_samples()

    # Step 6A: warn if sample count is too low for reliable FID.
    if num_samples < _MIN_RELIABLE_SAMPLES:
        warnings.warn(
            f"evaluate_sampler: num_samples={num_samples} is below the recommended "
            f"minimum of {_MIN_RELIABLE_SAMPLES} for reliable FID estimates. "
            "Use --full-eval or increase --num-samples for publication-quality results.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "FID reliability warning: num_samples=%d < %d. Results may be noisy.",
            num_samples, _MIN_RELIABLE_SAMPLES,
        )

    # Set RNG seed for reproducible sampling (Step 6C).
    if eval_cfg.seed is not None:
        torch.manual_seed(eval_cfg.seed)

    def _run_single_eval(seed_offset: int = 0) -> tuple[float, float, float, dict]:
        """Run one full evaluation pass; returns (fid, is_mean, is_std, extra_metrics)."""
        if eval_cfg.seed is not None:
            torch.manual_seed(eval_cfg.seed + seed_offset)

        # Phase 2 (2026-07-15) fix: FID was constructed with normalize=True while being
        # fed uint8 tensors. torchmetrics' update() does
        #     imgs = (imgs * 255).byte() if self.normalize else imgs
        # so a uint8 input is multiplied by 255 and WRAPS MOD 256, mapping pixel v to
        # (256 - v) -- i.e. every image is inverted before reaching Inception.
        # Because the same transform hits both real and fake, FID stayed a valid
        # distance and this is NOT why the pre-audit FIDs were ~326 (that was the
        # sampler, defect A1). But it computed FID in an inverted pixel space, so the
        # numbers are not comparable to published FashionMNIST FIDs -- which the paper
        # requires. uint8 inputs need normalize=False.
        # (Measured on identical inputs: 18.92 with the bug vs 20.79 correct.)
        fid_m = FrechetInceptionDistance(normalize=False).to(device)
        # InceptionScore is fed float [0,1] (images_for_isc), so normalize=True is
        # correct here -- do not "fix" this one to match.
        isc_m = InceptionScore(normalize=True).to(device)

        # Real images
        real_count = 0
        for real_batch, _ in loader:
            real_batch = real_batch.to(device)
            real_uint8 = ((real_batch.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
            if real_uint8.shape[1] == 1:
                real_uint8 = real_uint8.repeat(1, 3, 1, 1)
            fid_m.update(real_uint8, real=True)
            real_count += real_batch.shape[0]
            if real_count >= num_samples:
                break

        # Generated samples
        gen_count = 0
        nfe_total = 0.0
        control_energy_total = 0.0
        path_kl_total = 0.0
        jarzynski_total = 0.0
        t_start = time.time()
        while gen_count < num_samples:
            bs = min(eval_cfg.batch_size, num_samples - gen_count)
            sample, stats = simulator.sample(
                shape=(bs, channels, height, width),
                device=device,
                control=control_policy,
                return_stats=True,
            )
            gen_count += bs
            images = (sample.clamp(-1, 1) + 1.0) * 0.5  # [0,1]
            uint8_imgs = (images * 255.0).to(torch.uint8)
            if uint8_imgs.shape[1] == 1:
                uint8_imgs = uint8_imgs.repeat(1, 3, 1, 1)
                images_for_isc = images.repeat(1, 3, 1, 1)
            else:
                images_for_isc = images
            fid_m.update(uint8_imgs, real=False)
            isc_m.update(images_for_isc)
            nfe_total += stats.get("nfe_total", 0.0) * bs
            control_energy_total += stats.get("control_energy", 0.0)
            path_kl_total += stats.get("path_kl", 0.0) * bs
            jarzynski_total += stats.get("jarzynski", 0.0) * bs

        wall_clock = time.time() - t_start
        fid_val = float(fid_m.compute().item())
        is_mean_val, is_std_val = isc_m.compute()
        extra = {
            "nfe_per_sample": float(nfe_total / max(gen_count, 1)),
            "wall_clock_sec": float(wall_clock),
            "num_samples": gen_count,
        }
        if control_energy_total > 0:
            extra["control_energy_total"] = float(control_energy_total)
        if path_kl_total > 0:
            extra["path_kl"] = float(path_kl_total / max(gen_count, 1))
        if jarzynski_total != 0:
            extra["jarzynski"] = float(jarzynski_total / max(gen_count, 1))
        return fid_val, float(is_mean_val.item()), float(is_std_val.item()), extra

    fid_score, is_mean, is_std, extra = _run_single_eval(seed_offset=0)
    results: dict[str, float] = {
        "fid": fid_score,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        **extra,
    }

    # Step 6E: dual-seed FID variance when --full-eval.
    if eval_cfg.full_eval:
        fid_score2, _, _, _ = _run_single_eval(seed_offset=1)
        fid_mean = (fid_score + fid_score2) / 2.0
        fid_std = abs(fid_score - fid_score2) / 2.0  # half-range as std proxy
        results["fid_seed1"] = fid_score
        results["fid_seed2"] = fid_score2
        results["fid_mean"] = fid_mean
        results["fid_std"] = fid_std
        logger.info(
            "Full-eval dual-seed FID: seed1=%.2f seed2=%.2f mean=%.2f std=%.2f",
            fid_score, fid_score2, fid_mean, fid_std,
        )

    return results


def evaluate_ddpm_baseline(
    score_model: torch.nn.Module,
    sde_config: ScoreSDEConfig,
    eval_cfg: EvaluationConfig,
    baseline: str = "ddpm",
) -> dict[str, float]:
    """Run DDPM or DDIM baseline sampling and compute FID, IS, and NFE."""
    from its.samplers.ddpm import DDPMSampler, DDPMSamplerConfig

    ddpm_cfg = DDPMSamplerConfig(
        num_steps=sde_config.num_steps,
        use_ddim=baseline == "ddim",
        ddim_eta=0.0,
    )
    sampler = DDPMSampler(score_model, ddpm_cfg)
    device = torch.device(eval_cfg.device)
    loader, channels, height, width = _make_dataset(eval_cfg.dataset_name, eval_cfg.batch_size)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    isc = InceptionScore(normalize=True).to(device)
    num_samples = eval_cfg.resolved_num_samples()

    # Real images for FID
    real_count = 0
    for real_batch, _ in loader:
        real_batch = real_batch.to(device)
        real_uint8 = ((real_batch.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        if real_uint8.shape[1] == 1:
            real_uint8 = real_uint8.repeat(1, 3, 1, 1)
        fid.update(real_uint8, real=True)
        real_count += real_batch.shape[0]
        if real_count >= num_samples:
            break

    # Generated samples
    gen_count = 0
    nfe_total = 0.0
    while gen_count < num_samples:
        bs = min(eval_cfg.batch_size, num_samples - gen_count)
        sample, stats = sampler.sample((bs, channels, height, width), device=device)
        gen_count += bs
        images = (sample.clamp(-1, 1) + 1.0) * 0.5
        uint8_imgs = (images * 255.0).to(torch.uint8)
        if uint8_imgs.shape[1] == 1:
            uint8_imgs = uint8_imgs.repeat(1, 3, 1, 1)
            images_for_isc = images.repeat(1, 3, 1, 1)
        else:
            images_for_isc = images
        fid.update(uint8_imgs, real=False)
        isc.update(images_for_isc)
        nfe_total += stats.get("nfe_total", 0.0) * bs

    fid_score = float(fid.compute().item())
    is_mean, is_std = isc.compute()
    return {
        "fid": fid_score,
        "inception_score_mean": float(is_mean.item()),
        "inception_score_std": float(is_std.item()),
        "nfe_per_sample": float(nfe_total / max(gen_count, 1)),
    }


class Evaluator:
    """Convenience wrapper that holds evaluation configuration."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate(self, sampler: object, device: torch.device | None = None) -> dict:
        return evaluate_sampler(sampler, self.config, device=device)

    def evaluate_ddpm(
        self,
        score_model: torch.nn.Module,
        sde_config: "ScoreSDEConfig",
        baseline: str = "ddpm",
        device: torch.device | None = None,
    ) -> dict:
        return evaluate_ddpm_baseline(score_model, sde_config, self.config, baseline=baseline)
