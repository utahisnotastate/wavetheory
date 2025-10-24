"""
Neuro-Symbolic Integration Module
Combines PINN training with symbolic regression for Wave Theory discovery
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from .pinn_jax import WavePINN, PINNTrainer, create_pinn_model, compile_symbolic_to_jax
from .symbolic_engine import SymbolicRegressionEngine, PySRConfig, NeuroSymbolicOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WaveTheoryConfig:
    """Configuration for the complete Wave Theory system."""
    # Physics parameters
    G: float = 1.0
    wave_frequency: float = 0.5
    decay_length: float = 10.0
    dt: float = 0.01
    
    # PINN parameters
    pinn_epochs: int = 1000
    pinn_learning_rate: float = 0.001
    
    # Symbolic regression parameters
    sr_populations: int = 15
    sr_iterations: int = 100
    
    # Evolution parameters
    max_generations: int = 10
    convergence_threshold: float = 1e-4

class WaveTheorySystem:
    """
    Main system class that integrates all components.
    """
    
    def __init__(self, config: Optional[WaveTheoryConfig] = None):
        self.config = config or WaveTheoryConfig()
        
        # Initialize components
        self.pinn_model = None
        self.pinn_trainer = None
        self.symbolic_engine = None
        self.orchestrator = None
        
        # State
        self.current_equation = "F = -G * (m₁ * m₂ / r²) * sin(ωr) * exp(-r/λ)"
        self.generation = 1
        self.evolution_history = []
        
    def initialize(self, key: jax.random.PRNGKey):
        """Initialize all system components."""
        logger.info("Initializing Wave Theory system...")
        
        # Create PINN model
        self.pinn_model, self.pinn_trainer = create_pinn_model(key)
        
        # Create symbolic regression engine
        sr_config = PySRConfig(
            populations=self.config.sr_populations,
            niterations=self.config.sr_iterations
        )
        self.symbolic_engine = SymbolicRegressionEngine(sr_config)
        
        # Create orchestrator
        self.orchestrator = NeuroSymbolicOrchestrator(
            self.pinn_trainer,
            self.symbolic_engine,
            self  # This system acts as the simulation engine
        )
        
        logger.info("Wave Theory system initialized successfully")
    
    def run_simulation(self, steps: int = 1000, save_interval: int = 10) -> List[Dict]:
        """
        Run physics simulation and return data for training.
        This is a simplified simulation for demonstration.
        """
        logger.info(f"Running simulation for {steps} steps")
        
        # Initialize particles
        particles = [
            {'id': 0, 'position': [0, 0, 0], 'velocity': [0.5, 0, 0], 'mass': 5.0},
            {'id': 1, 'position': [10, 0, 0], 'velocity': [-0.5, 0.5, 0], 'mass': 5.0},
            {'id': 2, 'position': [5, 8.66, 0], 'velocity': [0, -0.5, 0], 'mass': 5.0}
        ]
        
        simulation_data = []
        
        for step in range(steps):
            # Calculate forces using current equation
            forces = [np.zeros(3) for _ in particles]
            
            for i in range(len(particles)):
                for j in range(len(particles)):
                    if i != j:
                        force = self._calculate_wave_force(particles[i], particles[j])
                        forces[i] += force
            
            # Update positions and velocities
            for i, particle in enumerate(particles):
                acceleration = forces[i] / particle['mass']
                particle['velocity'] = [v + a * self.config.dt for v, a in zip(particle['velocity'], acceleration)]
                particle['position'] = [p + v * self.config.dt for p, v in zip(particle['position'], particle['velocity'])]
            
            # Save data at intervals
            if step % save_interval == 0:
                energy = self._calculate_system_energy(particles)
                simulation_data.append({
                    'time': step * self.config.dt,
                    'bodies': [p.copy() for p in particles],
                    'energy': energy
                })
        
        logger.info(f"Simulation complete. Generated {len(simulation_data)} data points")
        return simulation_data
    
    def _calculate_wave_force(self, p1: Dict, p2: Dict) -> np.ndarray:
        """Calculate Wave Theory force between two particles."""
        r_vec = np.array(p2['position']) - np.array(p1['position'])
        r = np.linalg.norm(r_vec)
        
        if r < 1e-6:
            return np.zeros(3)
        
        magnitude = -self.config.G * (p1['mass'] * p2['mass'] / (r**2)) * \
                   np.sin(self.config.wave_frequency * r) * np.exp(-r / self.config.decay_length)
        
        return magnitude * (r_vec / r)
    
    def _calculate_system_energy(self, particles: List[Dict]) -> Dict[str, float]:
        """Calculate total system energy."""
        kinetic = 0.0
        potential = 0.0
        
        # Kinetic energy
        for p in particles:
            v_squared = sum(v**2 for v in p['velocity'])
            kinetic += 0.5 * p['mass'] * v_squared
        
        # Potential energy
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                r_vec = np.array(particles[j]['position']) - np.array(particles[i]['position'])
                r = np.linalg.norm(r_vec)
                if r > 1e-6:
                    potential += -self.config.G * particles[i]['mass'] * particles[j]['mass'] / r
        
        return {'kinetic': kinetic, 'potential': potential, 'total': kinetic + potential}
    
    def run_evolution(self, n_generations: Optional[int] = None) -> Dict[str, Any]:
        """Run the neuro-symbolic evolution process."""
        n_generations = n_generations or self.config.max_generations
        
        logger.info(f"Starting neuro-symbolic evolution for {n_generations} generations")
        
        results = self.orchestrator.run_evolution_cycle(n_generations)
        
        # Update system state
        if results['best_equation']:
            self.current_equation = results['best_equation']['equation']
            self.generation = results['final_generation']
            self.evolution_history = results['evolution_history']
        
        return results
    
    def get_current_equation(self) -> str:
        """Get the current discovered equation."""
        return self.current_equation
    
    def get_generation(self) -> int:
        """Get current generation number."""
        return self.generation
    
    def get_evolution_history(self) -> List[Dict]:
        """Get evolution history."""
        return self.evolution_history
    
    def update_physics_parameters(self, G: Optional[float] = None, 
                                 wave_frequency: Optional[float] = None,
                                 decay_length: Optional[float] = None):
        """Update physics parameters."""
        if G is not None:
            self.config.G = G
        if wave_frequency is not None:
            self.config.wave_frequency = wave_frequency
        if decay_length is not None:
            self.config.decay_length = decay_length
        
        logger.info(f"Updated physics parameters: G={self.config.G}, "
                   f"ω={self.config.wave_frequency}, λ={self.config.decay_length}")

# Factory function for easy initialization
def create_wave_theory_system(config: Optional[WaveTheoryConfig] = None, 
                            key: Optional[jax.random.PRNGKey] = None) -> WaveTheorySystem:
    """Create and initialize a Wave Theory system."""
    if key is None:
        key = jax.random.PRNGKey(42)
    
    system = WaveTheorySystem(config)
    system.initialize(key)
    return system
