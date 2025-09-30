"""
Advanced Physics Experiments and Presets
Pre-configured experiments for different physics scenarios
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class ExperimentType(Enum):
    """Types of physics experiments."""
    GRAVITATIONAL_WAVES = "gravitational_waves"
    QUANTUM_TUNNELING = "quantum_tunneling"
    CHAOTIC_DYNAMICS = "chaotic_dynamics"
    WAVE_INTERFERENCE = "wave_interference"
    PARTICLE_COLLISIONS = "particle_collisions"
    ORBITAL_MECHANICS = "orbital_mechanics"
    FIELD_RESONANCE = "field_resonance"
    THERMAL_DIFFUSION = "thermal_diffusion"

@dataclass
class PhysicsPreset:
    """Pre-configured physics experiment parameters."""
    name: str
    description: str
    experiment_type: ExperimentType
    particles: List[Dict[str, Any]]
    physics_params: Dict[str, float]
    simulation_params: Dict[str, Any]
    expected_behavior: str
    learning_objectives: List[str]

class PhysicsExperimentLibrary:
    """Library of pre-configured physics experiments."""
    
    def __init__(self):
        self.presets = self._create_presets()
    
    def _create_presets(self) -> Dict[str, PhysicsPreset]:
        """Create all physics experiment presets."""
        presets = {}
        
        # Gravitational Waves Experiment
        presets["gravitational_waves"] = PhysicsPreset(
            name="Gravitational Wave Detection",
            description="Simulate binary black hole merger creating gravitational waves",
            experiment_type=ExperimentType.GRAVITATIONAL_WAVES,
            particles=[
                {"position": [0, 0, 0], "velocity": [0, 0, 0], "mass": 20.0, "color": "#ff0000"},
                {"position": [15, 0, 0], "velocity": [0, 2.5, 0], "mass": 15.0, "color": "#0000ff"}
            ],
            physics_params={
                "G": 1.0,
                "wave_frequency": 0.1,
                "decay_length": 50.0,
                "damping": 0.01
            },
            simulation_params={
                "dt": 0.005,
                "duration": 100.0,
                "domain": {"x": [-50, 50], "y": [-50, 50], "z": [-50, 50]}
            },
            expected_behavior="Binary system should spiral inward, emitting gravitational waves",
            learning_objectives=[
                "Understand gravitational wave emission",
                "Observe orbital decay due to energy loss",
                "Learn about binary system dynamics"
            ]
        )
        
        # Quantum Tunneling Experiment
        presets["quantum_tunneling"] = PhysicsPreset(
            name="Quantum Tunneling Simulation",
            description="Particles tunneling through potential barriers",
            experiment_type=ExperimentType.QUANTUM_TUNNELING,
            particles=[
                {"position": [-20, 0, 0], "velocity": [1.0, 0, 0], "mass": 1.0, "color": "#00ff00"},
                {"position": [-15, 0, 0], "velocity": [0.8, 0, 0], "mass": 1.0, "color": "#00ff00"},
                {"position": [-10, 0, 0], "velocity": [1.2, 0, 0], "mass": 1.0, "color": "#00ff00"}
            ],
            physics_params={
                "G": 0.1,
                "wave_frequency": 2.0,
                "decay_length": 5.0,
                "barrier_strength": 0.5
            },
            simulation_params={
                "dt": 0.01,
                "duration": 50.0,
                "barrier_positions": [0, 5, 10]
            },
            expected_behavior="Some particles should tunnel through barriers",
            learning_objectives=[
                "Understand quantum tunneling probability",
                "Observe wave-particle duality",
                "Learn about potential barriers"
            ]
        )
        
        # Chaotic Dynamics Experiment
        presets["chaotic_dynamics"] = PhysicsPreset(
            name="Three-Body Chaotic System",
            description="Classic three-body problem showing chaotic behavior",
            experiment_type=ExperimentType.CHAOTIC_DYNAMICS,
            particles=[
                {"position": [0, 0, 0], "velocity": [0, 0, 0], "mass": 10.0, "color": "#ff0000"},
                {"position": [10, 0, 0], "velocity": [0, 1.5, 0], "mass": 5.0, "color": "#00ff00"},
                {"position": [5, 8.66, 0], "velocity": [0, -1.5, 0], "mass": 5.0, "color": "#0000ff"}
            ],
            physics_params={
                "G": 1.0,
                "wave_frequency": 0.3,
                "decay_length": 20.0,
                "chaos_factor": 0.1
            },
            simulation_params={
                "dt": 0.001,
                "duration": 200.0,
                "precision": "high"
            },
            expected_behavior="System should exhibit chaotic, unpredictable motion",
            learning_objectives=[
                "Understand chaotic dynamics",
                "Observe sensitive dependence on initial conditions",
                "Learn about three-body problem"
            ]
        )
        
        # Wave Interference Experiment
        presets["wave_interference"] = PhysicsPreset(
            name="Wave Interference Patterns",
            description="Multiple wave sources creating interference patterns",
            experiment_type=ExperimentType.WAVE_INTERFERENCE,
            particles=[
                {"position": [-10, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ff00ff"},
                {"position": [10, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#00ffff"},
                {"position": [0, 10, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ffff00"}
            ],
            physics_params={
                "G": 0.5,
                "wave_frequency": 1.0,
                "decay_length": 30.0,
                "interference_strength": 2.0
            },
            simulation_params={
                "dt": 0.01,
                "duration": 100.0,
                "wave_amplitude": 1.0
            },
            expected_behavior="Constructive and destructive interference patterns",
            learning_objectives=[
                "Understand wave interference",
                "Observe standing wave patterns",
                "Learn about wave superposition"
            ]
        )
        
        # Particle Collisions Experiment
        presets["particle_collisions"] = PhysicsPreset(
            name="High-Energy Particle Collisions",
            description="Simulate particle accelerator collisions",
            experiment_type=ExperimentType.PARTICLE_COLLISIONS,
            particles=[
                {"position": [-30, 0, 0], "velocity": [5.0, 0, 0], "mass": 2.0, "color": "#ff0000"},
                {"position": [30, 0, 0], "velocity": [-5.0, 0, 0], "mass": 2.0, "color": "#0000ff"},
                {"position": [-25, 2, 0], "velocity": [4.8, 0, 0], "mass": 1.5, "color": "#ff6600"},
                {"position": [25, -2, 0], "velocity": [-4.8, 0, 0], "mass": 1.5, "color": "#0066ff"}
            ],
            physics_params={
                "G": 2.0,
                "wave_frequency": 0.5,
                "decay_length": 10.0,
                "collision_energy": 10.0
            },
            simulation_params={
                "dt": 0.005,
                "duration": 30.0,
                "collision_detection": True
            },
            expected_behavior="High-energy collisions with particle creation",
            learning_objectives=[
                "Understand particle collisions",
                "Observe energy conservation",
                "Learn about collision dynamics"
            ]
        )
        
        # Orbital Mechanics Experiment
        presets["orbital_mechanics"] = PhysicsPreset(
            name="Planetary Orbital System",
            description="Multi-planet system with realistic orbital mechanics",
            experiment_type=ExperimentType.ORBITAL_MECHANICS,
            particles=[
                {"position": [0, 0, 0], "velocity": [0, 0, 0], "mass": 100.0, "color": "#ffff00"},  # Sun
                {"position": [20, 0, 0], "velocity": [0, 3.0, 0], "mass": 1.0, "color": "#00ff00"},   # Planet 1
                {"position": [35, 0, 0], "velocity": [0, 2.5, 0], "mass": 0.5, "color": "#0000ff"},  # Planet 2
                {"position": [50, 0, 0], "velocity": [0, 2.0, 0], "mass": 0.3, "color": "#ff6600"}   # Planet 3
            ],
            physics_params={
                "G": 1.0,
                "wave_frequency": 0.1,
                "decay_length": 100.0,
                "orbital_damping": 0.001
            },
            simulation_params={
                "dt": 0.01,
                "duration": 500.0,
                "orbital_precision": "high"
            },
            expected_behavior="Stable orbital motion with gravitational interactions",
            learning_objectives=[
                "Understand orbital mechanics",
                "Observe gravitational perturbations",
                "Learn about multi-body systems"
            ]
        )
        
        # Field Resonance Experiment
        presets["field_resonance"] = PhysicsPreset(
            name="Field Resonance and Standing Waves",
            description="Resonant field patterns in confined space",
            experiment_type=ExperimentType.FIELD_RESONANCE,
            particles=[
                {"position": [0, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ff00ff"},
                {"position": [10, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#00ffff"},
                {"position": [20, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ffff00"},
                {"position": [30, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ff6600"}
            ],
            physics_params={
                "G": 0.3,
                "wave_frequency": 0.8,
                "decay_length": 40.0,
                "resonance_frequency": 0.8
            },
            simulation_params={
                "dt": 0.01,
                "duration": 150.0,
                "boundary_conditions": "reflective"
            },
            expected_behavior="Standing wave patterns and resonance effects",
            learning_objectives=[
                "Understand field resonance",
                "Observe standing wave formation",
                "Learn about boundary conditions"
            ]
        )
        
        # Thermal Diffusion Experiment
        presets["thermal_diffusion"] = PhysicsPreset(
            name="Thermal Diffusion and Heat Transfer",
            description="Particles with temperature-dependent behavior",
            experiment_type=ExperimentType.THERMAL_DIFFUSION,
            particles=[
                {"position": [0, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ff0000", "temperature": 100.0},
                {"position": [5, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ff6600", "temperature": 80.0},
                {"position": [10, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ffff00", "temperature": 60.0},
                {"position": [15, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#00ff00", "temperature": 40.0},
                {"position": [20, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#0000ff", "temperature": 20.0}
            ],
            physics_params={
                "G": 0.2,
                "wave_frequency": 0.3,
                "decay_length": 15.0,
                "thermal_diffusivity": 0.1
            },
            simulation_params={
                "dt": 0.01,
                "duration": 200.0,
                "temperature_coupling": True
            },
            expected_behavior="Heat diffusion and temperature equilibration",
            learning_objectives=[
                "Understand thermal diffusion",
                "Observe heat transfer mechanisms",
                "Learn about temperature dynamics"
            ]
        )
        
        # Lighthouse-inspired: Blinking Universe Oscillator
        presets["blinking_universe"] = PhysicsPreset(
            name="Blinking Universe Oscillator",
            description="Gate interactions with a global blinking frequency to test stroboscopic stability",
            experiment_type=ExperimentType.FIELD_RESONANCE,
            particles=[
                {"position": [0, 0, 0], "velocity": [0.3, 0.2, 0], "mass": 2.0, "color": "#ff3366"},
                {"position": [8, 0, 0], "velocity": [0, -0.4, 0], "mass": 2.0, "color": "#33ccff"}
            ],
            physics_params={
                "G": 1.0,
                "wave_frequency": 0.5,
                "decay_length": 25.0,
                "blink_frequency": 1.0,  # gating frequency
                "blink_duty": 0.5        # fraction of time force is active
            },
            simulation_params={
                "dt": 0.005,
                "duration": 120.0,
                "stroboscopic": True
            },
            expected_behavior="Intermittent forcing may create resonant energy growth or stabilization",
            learning_objectives=[
                "Explore discrete-time forcing (blinking universe)",
                "Observe energy/stability under gated interactions",
                "Study resonance with blink frequency"
            ]
        )

        # Lighthouse-inspired: Gravity–EM Coupled Force
        presets["gravity_em_coupled"] = PhysicsPreset(
            name="Gravity–EM Coupled Force",
            description="Inverse-square attraction with sinusoidal modulation and exponential screening",
            experiment_type=ExperimentType.FIELD_RESONANCE,
            particles=[
                {"position": [0, 0, 0], "velocity": [0, 0, 0], "mass": 5.0, "color": "#ffaa00"},
                {"position": [12, 0, 0], "velocity": [0, 0.9, 0], "mass": 3.0, "color": "#00aaff"}
            ],
            physics_params={
                "G": 1.0,
                "wave_frequency": 0.6,   # omega in sin(omega r + phi)
                "decay_length": 18.0,    # screening length lambda
                "alpha": 0.2,            # modulation strength
                "phi": 0.0               # phase offset
            },
            simulation_params={
                "dt": 0.005,
                "duration": 200.0,
                "record_pareto": True
            },
            expected_behavior="Resonant separations show enhanced/ reduced attraction; quasi-stable orbits",
            learning_objectives=[
                "Test sinusoidal modulation of inverse-square forces",
                "Explore resonance and screening effects",
                "Compare to baseline gravitational case"
            ]
        )

        # Lighthouse-inspired: Phase-Matter Lattice
        presets["phase_matter_lattice"] = PhysicsPreset(
            name="Phase-Matter Lattice",
            description="Assign particle phases; interaction scales with cos(delta phase)",
            experiment_type=ExperimentType.WAVE_INTERFERENCE,
            particles=[
                {"position": [ -6, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#ff77ff", "phase": 0.0},
                {"position": [  0, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#77ff77", "phase": 2.094},
                {"position": [ +6, 0, 0], "velocity": [0, 0, 0], "mass": 1.0, "color": "#7777ff", "phase": 4.188}
            ],
            physics_params={
                "G": 0.6,
                "wave_frequency": 1.2,
                "decay_length": 30.0,
                "phase_coupling": 1.0
            },
            simulation_params={
                "dt": 0.01,
                "duration": 160.0,
                "lattice_spacing": 6.0
            },
            expected_behavior="Phase alignment and segregation; interference-like clustering",
            learning_objectives=[
                "Study phase-dependent interactions",
                "Relate to interference and superposition",
                "Explore emergent lattice behavior"
            ]
        )

        # Lighthouse-inspired: Time Dilation Sweep
        presets["time_dilation_sweep"] = PhysicsPreset(
            name="Time Dilation Sweep",
            description="Position-dependent time scaling s(x)=1+beta·sin(kx) applied to integrator",
            experiment_type=ExperimentType.CHAOTIC_DYNAMICS,
            particles=[
                {"position": [ -10, 0, 0], "velocity": [0.8,  0.2, 0], "mass": 1.5, "color": "#cc4444"},
                {"position": [  10, 0, 0], "velocity": [-0.8, -0.2, 0], "mass": 1.5, "color": "#44cc44"}
            ],
            physics_params={
                "G": 0.8,
                "wave_frequency": 0.4,
                "decay_length": 22.0,
                "beta": 0.3,
                "k": 0.3
            },
            simulation_params={
                "dt": 0.005,
                "duration": 120.0,
                "variable_timestep": True
            },
            expected_behavior="Regions of slowed/accelerated dynamics; potential trapping",
            learning_objectives=[
                "Experiment with variable time-stepping",
                "Observe energy artifacts vs corrected schemes",
                "Compare trajectories with uniform dt"
            ]
        )

        return presets
    
    def get_preset(self, name: str) -> Optional[PhysicsPreset]:
        """Get a specific physics preset."""
        return self.presets.get(name)
    
    def list_presets(self) -> List[Dict[str, str]]:
        """List all available presets."""
        return [
            {
                "name": name,
                "title": preset.name,
                "description": preset.description,
                "type": preset.experiment_type.value
            }
            for name, preset in self.presets.items()
        ]
    
    def get_presets_by_type(self, experiment_type: ExperimentType) -> List[PhysicsPreset]:
        """Get all presets of a specific type."""
        return [
            preset for preset in self.presets.values()
            if preset.experiment_type == experiment_type
        ]
    
    def create_custom_experiment(self, name: str, description: str, 
                               particles: List[Dict], physics_params: Dict,
                               simulation_params: Dict) -> PhysicsPreset:
        """Create a custom physics experiment."""
        return PhysicsPreset(
            name=name,
            description=description,
            experiment_type=ExperimentType.CHAOTIC_DYNAMICS,  # Default type
            particles=particles,
            physics_params=physics_params,
            simulation_params=simulation_params,
            expected_behavior="Custom experiment behavior",
            learning_objectives=["Custom learning objectives"]
        )
    
    def save_experiment(self, preset: PhysicsPreset, filename: str) -> None:
        """Save an experiment to file."""
        data = {
            "name": preset.name,
            "description": preset.description,
            "experiment_type": preset.experiment_type.value,
            "particles": preset.particles,
            "physics_params": preset.physics_params,
            "simulation_params": preset.simulation_params,
            "expected_behavior": preset.expected_behavior,
            "learning_objectives": preset.learning_objectives
        }
        
        with open(f"data/experiments/{filename}.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def load_experiment(self, filename: str) -> Optional[PhysicsPreset]:
        """Load an experiment from file."""
        try:
            with open(f"data/experiments/{filename}.json", "r") as f:
                data = json.load(f)
            
            return PhysicsPreset(
                name=data["name"],
                description=data["description"],
                experiment_type=ExperimentType(data["experiment_type"]),
                particles=data["particles"],
                physics_params=data["physics_params"],
                simulation_params=data["simulation_params"],
                expected_behavior=data["expected_behavior"],
                learning_objectives=data["learning_objectives"]
            )
        except FileNotFoundError:
            return None

# Global experiment library
experiment_library = PhysicsExperimentLibrary()
