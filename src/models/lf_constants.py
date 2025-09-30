"""
Lighthouse Frequency (LF) support utilities.
Implements time-varying blinking function B(t), G(t), and c(t).
All functions are pure and JAX-friendly when JAX is available; fall back to NumPy.
"""
from typing import Tuple

try:
    import jax.numpy as jnp
    Array = jnp.ndarray
except Exception:  # pragma: no cover
    import numpy as jnp  # type: ignore
    Array = jnp.ndarray  # type: ignore


def temporal_blinking_function(t: Array, f_lf_hz: float, tau: float = 1e-17) -> Array:
    """Smooth blinking function B(t) in [0,1].
    Uses a fast periodic sigmoid envelope to avoid discontinuities.
    - f_lf_hz: Lighthouse frequency (e.g., ~1e12 Hz)
    - tau: transition sharpness (seconds)
    """
    # phase in radians
    phase = 2.0 * jnp.pi * f_lf_hz * t
    # use squared sine into logistic for sharp but smooth gating
    s = jnp.sin(phase)
    # logistic gate centered at |s| near 0 with steepness 1/tau scaled
    k = 1.0 / jnp.maximum(tau, 1e-21)
    gate = 1.0 / (1.0 + jnp.exp(-k * (jnp.abs(s) - 0.5)))
    # normalize to ~[0,1]
    return 1.0 - gate


def G_of_t(t: Array, G_avg: float, G_amp: float, f_lf_hz: float, phi: float = 0.0) -> Array:
    """Time-varying gravitational constant.
    G(t) = G_avg * (1 + (G_amp/G_avg) * sin(2π f t + phi))
    """
    return G_avg * (1.0 + (G_amp / jnp.maximum(G_avg, 1e-30)) * jnp.sin(2.0 * jnp.pi * f_lf_hz * t + phi))


def c_of_t(t: Array, c0: float, f_lf_hz: float, phi: float = 0.0, drift_per_year: float = 0.0, years: float = 0.0) -> Array:
    """Time-varying speed of light.
    c(t) = c0 * (1 + eps * sin(2π f t + phi)) + drift_per_year * years
    The sinusoidal amplitude eps can be implicit via physical calibration; use small default 1e-9.
    """
    eps = 1e-9
    return c0 * (1.0 + eps * jnp.sin(2.0 * jnp.pi * f_lf_hz * t + phi)) + drift_per_year * years


def lf_force_gate(t: Array, f_lf_hz: float, tau: float = 1e-17, duty: float = 0.5) -> Array:
    """Convenience gate in [0,1] blending B(t) toward desired duty cycle."""
    B = temporal_blinking_function(t, f_lf_hz, tau)
    # scale to match duty (mean(B) -> duty) via linear normalization
    mean_target = duty
    mean_B = 0.5  # approximate; avoids heavy reductions
    scale = (mean_target / jnp.maximum(mean_B, 1e-6))
    return jnp.clip(B * scale, 0.0, 1.0)
