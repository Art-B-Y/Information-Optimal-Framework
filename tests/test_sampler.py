import numpy as np

from its.controllers import ControlConfig, build_control_policy
from its.samplers import ControlledSDEConfig, ControlledSampler, PotentialConfig


def test_controlled_sampler_shapes() -> None:
    sampler = ControlledSampler(
        ControlledSDEConfig(dt=0.05, steps=12, diffusion=0.5, control_scale=0.3),
        PotentialConfig(kind="harmonic", dim=1),
    )
    controller = build_control_policy(ControlConfig(state_dim=1, hidden_dim=16, depth=2))
    controller.eval()
    result = sampler.simulate(np.zeros((1,), dtype=np.float32), controller=controller, seed=42)

    assert result["states"].shape == (13, 1)
    assert result["controls"].shape == (12, 1)
    assert result["control_torch"].shape == (12, 1)
    assert np.isfinite(result["control_energy"]).all()
