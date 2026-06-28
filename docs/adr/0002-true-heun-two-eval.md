# Faithful two-evaluation true-Heun reverse step, not diffusers' single-call idiom

The Scheduler's reverse step is a true trapezoidal Heun: the Pipeline evaluates the
UNet **twice per timestep** (at `z_t` and at the Euler-advanced `z_{t+dt}`) and the
Scheduler averages the two velocities. We rejected diffusers' single-call
`step(model_output, t, sample)` multistep idiom — the canonical `for t: model();
step()` loop — even though it is the most "diffusers-style" shape, because a true
Heun *requires* the derivative at the advanced point. A single-call scheme that
reuses stored history is a *different* 2nd-order method and would not bit-reproduce
`hope/sampling/x0.py`, breaking reproducibility against already-trained JiT
checkpoints — the whole point of migrating a working sampler.

## Consequences

- The Pipeline denoise loop is `pred = model(z_t); z_euler = scheduler.euler_step(...);
  corr = model(z_euler); z = scheduler.heun_correct(v1, v2, dt)`, not the single
  `model(); scheduler.step()`.
- The Scheduler exposes a two-phase reverse API (predictor + corrector, or a `step`
  that consumes both model outputs) rather than one `step()`; this is the price of
  correctness and is documented inline.
