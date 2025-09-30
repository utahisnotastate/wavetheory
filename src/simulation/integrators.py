"""
Adaptive leapfrog integrator with time-varying constants support.
This keeps a velocity Verlet-style scheme but calls acceleration(t, x, v)
that can incorporate LF-modulated G(t), c(t), and gating B(t).
"""
from typing import Callable, Tuple
import numpy as np

State = Tuple[np.ndarray, np.ndarray]  # (x, v)


def leapfrog_adaptive(
    state: State,
    t: float,
    dt: float,
    acceleration_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    dt_min: float = 1e-18,
    dt_max: float = 1e-3,
    adapt_on: bool = True,
    adapt_safety: float = 0.5,
) -> Tuple[State, float]:
    """Single adaptive step.
    - acceleration_fn returns a(t, x, v) respecting time-varying constants.
    - crude adaptation based on |a| and a finite-difference jerk estimate.
    Returns: (new_state, suggested_dt)
    """
    x, v = state
    a0 = acceleration_fn(t, x, v)

    # half-step velocity
    v_half = v + 0.5 * dt * a0
    # full-step position
    x_new = x + dt * v_half
    # acceleration at new time/position
    a1 = acceleration_fn(t + dt, x_new, v_half)
    # full-step velocity
    v_new = v_half + 0.5 * dt * a1

    # adaptive dt suggestion (very conservative):
    if adapt_on:
        a_mag = np.linalg.norm(a1) + 1e-30
        jerk_est = np.linalg.norm((a1 - a0) / max(dt, 1e-30))
        denom = a_mag + np.sqrt(jerk_est) + 1e-30
        candidate = adapt_safety * 1.0 / denom
        dt_suggest = float(np.clip(candidate, dt_min, dt_max))
    else:
        dt_suggest = dt

    return (x_new, v_new), dt_suggest
