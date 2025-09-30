# Lighthouse Frequency (LF) Integration Blueprint

This document explains the minimal integration added:

## Modules
- `src/models/lf_constants.py`
  - `temporal_blinking_function(t, f_lf_hz, tau)` → smooth B(t)
  - `G_of_t(t, G_avg, G_amp, f_lf_hz, phi)` → time-varying G(t)
  - `c_of_t(t, c0, f_lf_hz, phi, drift_per_year, years)` → c(t)
  - `lf_force_gate(t, f_lf_hz, tau, duty)` → convenience gate
- `src/models/particle1024.py` → 1024-QAM-inspired particle state with coherence and collapse
- `src/simulation/integrators.py` → `leapfrog_adaptive` supporting explicit time dependence

## Config
`configs/config.yaml`
```yaml
physics:
  lighthouse_model:
    enabled: false
    lf_frequency_hz: 1.1e12
    blink_tau: 1e-17
    blink_duty: 0.5
    G_amp: 0.0
    phi: 0.0
    c_drift_per_year: 0.0
```

Turn on by setting `enabled: true` and using these values in your acceleration function.

## Usage Sketch
```python
from models.lf_constants import G_of_t, lf_force_gate
from simulation.integrators import leapfrog_adaptive

# Define acceleration that consults LF config
def acceleration(t, x, v):
    Gt = G_of_t(t, G_avg=1.0, G_amp=cfg.G_amp, f_lf_hz=cfg.lf_frequency_hz)
    gate = lf_force_gate(t, cfg.lf_frequency_hz, cfg.blink_tau, cfg.blink_duty)
    a = compute_pairwise_gravity(x, Gt) * gate
    return a

state, dt = (x0, v0), 1e-16
for _ in range(steps):
    state, dt = leapfrog_adaptive(state, t, dt, acceleration)
    t += dt
```

## Notes
- Energy is not conserved by design when LF is enabled; audit energy inflow/outflow via source terms.
- Start with tens to hundreds of femtoseconds; THz cycles are expensive.
- The presets `blinking_universe` etc. provide sandbox scenarios aligned with LF ideas.
