"""
Particle1024: 1024-QAM-inspired discrete particle state.
Encodes 10-bit symbol index mapped to (I,Q) constellation with 32x32 grid.
Includes coherence metric and simple collapse method.
"""
from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass
class Particle1024:
    symbol_index: int
    position: np.ndarray  # shape (3,)
    velocity: np.ndarray  # shape (3,)
    mass_state_q: float   # quantized amplitude level (maps to mass)
    charge_phase_q: float # quantized phase (radians)
    constellation_coords: Tuple[float, float]
    binary_state: str     # 10-bit string
    coherence_metric: float = 1.0

    @staticmethod
    def from_symbol(symbol_index: int, position, velocity) -> "Particle1024":
        assert 0 <= symbol_index < 1024
        i = symbol_index % 32
        q = symbol_index // 32
        # map indices to normalized levels [-1,1]
        I = -1.0 + 2.0 * i / 31.0
        Q = -1.0 + 2.0 * q / 31.0
        mass = np.interp(np.hypot(I, Q), [0.0, np.sqrt(2)], [1.0, 32.0])  # 32 discrete-ish levels
        phase = np.arctan2(Q, I)
        bits = format(symbol_index, "010b")
        return Particle1024(
            symbol_index=symbol_index,
            position=np.asarray(position, dtype=float),
            velocity=np.asarray(velocity, dtype=float),
            mass_state_q=float(mass),
            charge_phase_q=float(phase),
            constellation_coords=(float(I), float(Q)),
            binary_state=bits,
            coherence_metric=1.0,
        )

    def measure_coherence(self, I_meas: float, Q_meas: float, noise_sigma: float = 0.05) -> float:
        I0, Q0 = self.constellation_coords
        d = np.hypot(I_meas - I0, Q_meas - Q0)
        # map distance to [0,1] where 1 is perfect coherence
        self.coherence_metric = float(np.exp(-0.5 * (d / max(noise_sigma, 1e-6)) ** 2))
        return self.coherence_metric

    def collapse(self, rng: np.random.Generator, threshold: float = 0.5) -> None:
        if self.coherence_metric >= threshold:
            return
        # choose nearest constellation point (greedy WFC analogue)
        I0, Q0 = self.constellation_coords
        i = int(round(31 * (I0 + 1.0) / 2.0))
        q = int(round(31 * (Q0 + 1.0) / 2.0))
        i = int(np.clip(i + rng.integers(-1, 2), 0, 31))
        q = int(np.clip(q + rng.integers(-1, 2), 0, 31))
        new_symbol = q * 32 + i
        updated = Particle1024.from_symbol(new_symbol, self.position, self.velocity)
        self.__dict__.update(updated.__dict__)
