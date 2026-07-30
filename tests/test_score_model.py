import torch

from its.models import ScoreUNetConfig, build_score_model
from its.sde import ScoreSDEConfig, ScoreSDESimulator


def test_score_model_forward_pass() -> None:
    config = ScoreUNetConfig(in_channels=3, base_channels=32, channel_mults=(1, 2))
    model = build_score_model(config)
    x = torch.randn(4, 3, 32, 32)
    sigma = torch.full((4, 1, 1, 1), 0.1)
    out = model(x, sigma)
    assert out.shape == x.shape


def test_score_sde_sampling_runs() -> None:
    config = ScoreUNetConfig(in_channels=3, base_channels=16, channel_mults=(1,))
    model = build_score_model(config)
    sde_cfg = ScoreSDEConfig(num_steps=10, beta_min=0.1, beta_max=0.5)
    simulator = ScoreSDESimulator(model, sde_cfg)
    samples = simulator.sample((2, 3, 32, 32), device=torch.device("cpu"))
    assert samples.shape == (2, 3, 32, 32)
