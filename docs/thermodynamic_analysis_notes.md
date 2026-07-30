# Thermodynamic Analysis: Controller Collapse as Non-Equilibrium Phase Transition

## 1. Physical Interpretation of the Collapse Attractor

The controlled SDE can be interpreted through non-equilibrium statistical mechanics. The uncontrolled reverse diffusion follows the Langevin dynamics:

    dx_t = -0.5 * beta_t * (x_t + score(x_t)) dt + sqrt(beta_t) dW_t

Adding a control signal u_theta modifies the drift, changing the path measure from P_0 (uncontrolled) to P_u (controlled). The Girsanov path KL divergence quantifies the thermodynamic cost:

    KL(P_u || P_0) = E[W_irr / kT]

where W_irr is the irreversible work done by the control force over the trajectory. The control signal effectively plays the role of an external non-equilibrium driving force. Training to minimize path KL is equivalent to minimizing irreversible work — the controller learns to produce the most thermodynamically reversible perturbation, which is zero perturbation.

This is the non-equilibrium generalization of detailed balance: a system that minimizes irreversible work at all times returns to the unperturbed trajectory. The collapse is not a numerical artifact but a thermodynamic principle — the optimizer has found the globally reversible (zero work) solution.

## 2. Jarzynski Equality and the Free Energy Bound

The Jarzynski equality connects the irreversible work distribution to the free energy difference:

    <exp(-W_irr / kT)> = exp(-Delta_F / kT)

In the ITS framework, the Jarzynski estimator approximates the free energy difference between the target distribution P_data and the diffusion prior. At collapse (u_theta = 0), the Jarzynski estimator converges to the free energy of the uncontrolled process, providing no information about the target distribution. The controller has failed to learn any reduction in free energy.

The diagnostic observation that path KL oscillates around zero (mean = -0.002, std = 0.013) at epoch 50 is consistent with this: the near-zero control produces a path measure nearly identical to the uncontrolled process, giving KL ≈ 0. The Jarzynski estimator at collapse provides no signal distinguishing the prior from the target.

## 3. The Collapse as an Entropy Production Crisis

In the language of stochastic thermodynamics, the entropy production rate of the controlled process is:

    sigma = sum_t (0.5 * ||u_t||^2 * dt)

which equals the control energy. At collapse, sigma → 0, meaning the controller has driven the entropy production to its minimum (the uncontrolled baseline). This is analogous to a system at minimum dissipation, which by the minimum entropy production principle corresponds to a steady state near equilibrium.

The collapse can be interpreted as a **non-equilibrium phase transition**: as training progresses, the system transitions from a high-dissipation regime (large control signals, high path KL) to a low-dissipation equilibrium-like state (zero control, zero path KL). The training objective, by penalizing path KL and control energy simultaneously, creates an energy landscape with a single attractive fixed point at u = 0.

## 4. Implications for the v2 Redesign

The v2 objective removes the thermodynamic attractors while preserving the information-theoretic framework:

1. **Detaching path KL gradient**: The path KL is still *measured* (for monitoring) but does not provide gradient signal to the controller. This breaks the irreversible-work minimization attractor.

2. **REINFORCE quality signal**: Rather than differentiating through the SDE trajectory (BPTT, which suffers gradient vanishing over 50 steps), the REINFORCE estimator uses the policy gradient theorem:

        nabla_theta J = E_tau[R(tau) * nabla_theta log p(tau | theta)]

    where R(tau) is the quality reward and nabla_theta log p(tau | theta) = sum_t nabla_theta log_rn_t. This provides an unbiased gradient estimate that does not require differentiating through the SDE forward pass.

3. **Warmup schedule**: The path KL weight starts at zero, preventing the early collapse (which occurs at epoch 5-10 in the collapsed run). During warmup, the controller receives only quality-based gradient signal, allowing it to establish non-zero output before the thermodynamic penalty is introduced.

The redesigned objective can be understood thermodynamically as: optimize the trajectory distribution to be close to the data distribution (quality) while allowing bounded irreversible work (path KL with small weight). This is analogous to optimally controlled non-equilibrium processes that minimize dissipation while achieving a target work output — the paradigm of stochastic thermodynamics optimal control.
