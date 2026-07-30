from its.experiments import build_double_well_experiment
from its.training import run_training


def test_training_loop_executes() -> None:
    bundle = build_double_well_experiment(device="cpu")
    bundle["training"].steps = 3
    bundle["training"].sampler_steps_initial = 30
    bundle["training"].sampler_steps_final = 30
    bundle["training"].temperature_initial = bundle["sampler"].config.temperature
    bundle["training"].temperature_final = bundle["sampler"].config.temperature
    bundle["training"].log_diagnostics_every = 1
    bundle["training"].occupancy_balance_weight = 0.0
    bundle["training"].occupancy_balance_weight_final = 0.0
    bundle["training"].y0_noise_std = 0.0
    bundle["training"].batch_size = 1
    bundle["training"].enforce_symmetry = False
    bundle["sampler"].config.steps = 30
    bundle["sampler"].config.dt = 0.01
    metrics = run_training(
        sampler=bundle["sampler"],
        controller=bundle["controller"],
        config=bundle["training"],
        quality_fn=bundle.get("quality_fn"),
    )
    assert "loss" in metrics
