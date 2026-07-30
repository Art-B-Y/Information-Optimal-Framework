# Controller Collapse Analysis: Session 10 Root Cause Diagnosis

## Summary

Session 9 revealed that the ConvControlPolicy trained with the legacy objective collapses to near-zero output. At epoch 50, the controller magnitude is **0.061% of the score network magnitude** — effectively the network learned to output zero, making it functionally identical to the uncontrolled DDPM/DDIM baseline.

## Quantitative Evidence

### Collapse Trajectory (controlled_conv_seed42)

| Epoch | ||u_theta|| | ||score|| | Ratio (%) |
|-------|------------|-----------|-----------|
| 5     | 0.6038     | 18.065    | 3.34%     |
| 10    | 0.0527     | 18.065    | 0.29%     |
| 15    | 0.2478     | 18.065    | 1.37%     |
| 20    | 0.0858     | 18.065    | 0.47%     |
| 25    | 0.0929     | 18.065    | 0.51%     |
| 30    | 0.0677     | 18.065    | 0.37%     |
| 35    | 0.0505     | 18.065    | 0.28%     |
| 40    | 0.0283     | 18.065    | 0.16%     |
| 45    | 0.0183     | 18.065    | 0.10%     |
| 50    | 0.0110     | 18.065    | **0.061%** |

The controller oscillates briefly at epoch 5 then begins a near-monotonic collapse. By epoch 50, control magnitude is 55x lower than at epoch 5.

### Loss Term Magnitudes at Epoch 50

| Term | Mean | Std | Observation |
|------|------|-----|-------------|
| DSM loss | 0.726 | 0.045 | Stable — score model is frozen and well-trained |
| Control energy | 9.3e-7 | 3e-8 | Essentially zero — confirmed collapse |
| Path KL | -0.002 | 0.013 | Oscillating around 0 — collapsed controller = minimum KL |
| Quality loss | 2.62 | 0.42 | High and variable — feature matching fails to improve |

### Gradient Flow Diagnosis (Session 10 Step 1B)

Gradient magnitudes on the controller output layer when each loss term is backpropagated individually:

| Loss Term | Output Layer Gradient | First Conv Gradient |
|-----------|----------------------|---------------------|
| DSM loss | 0.000 | 0.000 |
| Control energy (×0.01) | 8.1e-6 | 1.1e-10 |
| **Path KL (×0.1)** | **1.45** | **2.5e-5** |
| Quality loss (×0.01) | 0.000 | 0.000 |

**Diagnosis: `PATH_KL_DOMINANT_QUALITY_BROKEN`**

## Root Cause: Two Independent Failure Modes

### Failure Mode 1: Path KL Dominates (Primary Cause)

The Girsanov path KL penalty is:

    KL(P_controlled || P_uncontrolled) = E[sum_t (0.5 ||u_t||^2 * dt - u_t . dW_t)]

This is **minimized when u_t = 0** for all t. The gradient of path KL with respect to controller parameters is 1.45 on the output layer — 180,000x larger than the quality gradient. The path KL penalty acts as a direct gradient attractor pushing the controller to zero.

### Failure Mode 2: Quality Gradient Does Not Flow (Secondary Cause)

The quality loss (feature matching via Inception-v3) has **zero gradient** reaching the controller output layer. The gradient chain breaks across:
1. The 50-step SDE simulation (vanishing gradients through long recurrent computation)
2. The Inception-v3 network (299×299 resize + deep CNN)

This means quality loss can never overcome the path KL attractor, regardless of its weight.

### Non-Factor: Control Energy

Control energy gradient (8.1e-6) is negligible compared to path KL (1.45). The `control_weight=0.01` scaling makes it 170,000x weaker than path KL.

## Why Zero Is a Global Optimum

Given the legacy objective:

    L = DSM + 0.01 * ||u||^2 + 0.1 * PathKL + 0.01 * Quality

- DSM has no gradient to controller (score model frozen)
- Quality has zero gradient to controller (broken chain)
- Both CE and PathKL are minimized at u=0

The controller has **no gradient signal** toward non-zero outputs and **strong gradient signal** toward zero. The collapse is not a training failure — it is the mathematically correct optimizer behavior for the legacy objective.

## FID Implication

FID of the collapsed controller (327.47 ± 1.57) matches the DDPM baseline precisely because the controller output at near-zero makes the controlled SDE converge to the same distribution as the uncontrolled DDPM. The similarity in FID across all 3 seeds (seed 42: 328.0, seed 1: 327.5, seed 7: 327.0) confirms this is deterministic, not stochastic failure.

## Session 10 Fixes

The v2 objective redesign addresses both failure modes:

1. **Detach control energy** — removes the zero attractor
2. **Detach path KL or use minimal weight (0.01)** — reduces the dominant collapse signal
3. **REINFORCE quality signal** — bypasses the broken gradient chain; rewards are per-sample feature distances, policy update is policy gradient not BPTT
4. **WarmupSchedule** — path KL weight starts at 0, ramps over 5 epochs, preventing early collapse
5. **Scaled random init** — breaks zero symmetry in output layer, prevents converging back immediately
6. **TwoPhaseScheduler** — very small LR for 5 epochs lets quality signal establish direction before path KL kicks in

See `scripts/train_redesigned_v2.py` for the full training configuration.
