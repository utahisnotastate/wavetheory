"""
PySR Integration for Symbolic Regression with Quality-Diversity
Advanced symbolic discovery engine for the Wave Theory system
"""

import numpy as np
import pandas as pd
from pysr import PySRRegressor
import sympy as sp
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass, asdict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# PYSR CONFIGURATION AND WRAPPER
# =====================================================================

@dataclass
class PySRConfig:
    """Configuration for PySR symbolic regression."""
    populations: int = 15
    population_size: int = 33
    maxsize: int = 20
    maxdepth: int = 10
    niterations: int = 100
    ncyclesperiteration: int = 550
    
    # Operators
    binary_operators: List[str] = None
    unary_operators: List[str] = None
    
    # Quality-Diversity settings
    use_frequency: bool = True
    use_frequency_in_tournament: bool = True
    adaptive_parsimony_scaling: float = 20.0
    parsimony: float = 0.0032
    
    # Advanced features
    batching: bool = True
    batch_size: int = 50
    denoise: bool = True
    select_k_features: int = None
    
    # Optimization
    optimizer_algorithm: str = "BFGS"
    optimizer_nrestarts: int = 2
    optimize_probability: float = 0.14
    optimizer_iterations: int = 8
    
    # Constraints and priors
    constraints: Dict = None
    loss: str = "L2DistLoss()"
    
    # Performance
    procs: int = 4
    multithreading: bool = True
    cluster_manager: Optional[str] = None
    
    # Output
    equation_file: str = "wave_theory_equations.csv"
    verbosity: int = 1
    progress: bool = True
    
    def __post_init__(self):
        if self.binary_operators is None:
            self.binary_operators = ["+", "-", "*", "/", "^"]
        if self.unary_operators is None:
            self.unary_operators = ["sin", "cos", "exp", "log", "sqrt", "tanh"]


class SymbolicRegressionEngine:
    """
    Advanced symbolic regression engine using PySR with quality-diversity optimization.
    """
    
    def __init__(self, config: Optional[PySRConfig] = None):
        self.config = config or PySRConfig()
        self.regressor = None
        self.best_equations = []
        self.pareto_front = []
        self.generation = 0
        
        # Initialize PySR
        self._initialize_regressor()
    
    def _initialize_regressor(self):
        """Initialize PySR with configured settings."""
        self.regressor = PySRRegressor(
            # Model selection
            model_selection="best",
            
            # Search parameters
            populations=self.config.populations,
            population_size=self.config.population_size,
            maxsize=self.config.maxsize,
            maxdepth=self.config.maxdepth,
            niterations=self.config.niterations,
            ncycles_per_iteration=self.config.ncyclesperiteration,
            
            # Operators
            binary_operators=self.config.binary_operators,
            unary_operators=self.config.unary_operators,
            
            # Complexity and parsimony
            parsimony=self.config.parsimony,
            adaptive_parsimony_scaling=self.config.adaptive_parsimony_scaling,
            
            # Quality-diversity
            use_frequency=self.config.use_frequency,
            use_frequency_in_tournament=self.config.use_frequency_in_tournament,
            
            # Optimization
            optimizer_algorithm=self.config.optimizer_algorithm,
            optimizer_nrestarts=self.config.optimizer_nrestarts,
            optimize_probability=self.config.optimize_probability,
            optimizer_iterations=self.config.optimizer_iterations,
            
            # Performance
            procs=self.config.procs,
            multithreading=self.config.multithreading,
            
            # Output
            equation_file=self.config.equation_file,
            verbosity=self.config.verbosity,
            progress=self.config.progress,
            
            # Advanced
            batching=self.config.batching,
            batch_size=self.config.batch_size,
            denoise=self.config.denoise,
            
            # Loss
            loss=self.config.loss,
            
            # Random state for reproducibility
            random_state=42,
            
            # Warm start capability
            warm_start=True
        )
        
        logger.info("PySR regressor initialized with quality-diversity optimization")
    
    def prepare_physics_data(self, simulation_data: List[Dict], 
                            pinn_residuals: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Prepare data from physics simulation for symbolic regression.
        
        Args:
            simulation_data: List of state dictionaries from simulation
            pinn_residuals: Optional residuals from PINN to guide search
        
        Returns:
            DataFrame with features and targets for regression
        """
        features = []
        targets = []
        
        for i, state in enumerate(simulation_data[:-1]):
            next_state = simulation_data[i + 1]
            
            for body in state['bodies']:
                # Extract features
                r = np.linalg.norm(body['position'])
                v = np.linalg.norm(body['velocity'])
                
                # Calculate accelerations (as proxy for forces)
                next_body = next((b for b in next_state['bodies'] if b['id'] == body['id']), None)
                if next_body:
                    dt = next_state['time'] - state['time']
                    if dt > 0:
                        acc = (np.array(next_body['velocity']) - np.array(body['velocity'])) / dt
                        force_magnitude = np.linalg.norm(acc) * body['mass']
                        
                        # Features: [r, v, m, t, energy]
                        features.append([
                            r,
                            v,
                            body['mass'],
                            state['time'],
                            state['energy']['total']
                        ])
                        
                        # Target: force magnitude (or residual if provided)
                        if pinn_residuals is not None and i < len(pinn_residuals):
                            targets.append(pinn_residuals[i])
                        else:
                            targets.append(force_magnitude)
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=['r', 'v', 'mass', 'time', 'energy'])
        df['target'] = targets
        
        # Add derived features that might be useful
        df['r_squared'] = df['r'] ** 2
        df['r_inv'] = 1.0 / (df['r'] + 1e-6)
        df['momentum'] = df['mass'] * df['v']
        df['kinetic_energy'] = 0.5 * df['mass'] * df['v'] ** 2
        
        logger.info(f"Prepared {len(df)} data points for symbolic regression")
        return df
    
    def discover_equation(self, X: pd.DataFrame, y: np.ndarray, 
                         feature_names: Optional[List[str]] = None,
                         warm_start_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Run symbolic regression to discover equations.
        
        Returns:
            Dictionary containing discovered equations and metrics
        """
        self.generation += 1
        logger.info(f"Starting symbolic discovery - Generation {self.generation}")
        
        if feature_names is None:
            feature_names = list(X.columns)
        
        # Warm start from previous best if available
        if warm_start_from and hasattr(self.regressor, 'equations_'):
            logger.info("Warm starting from previous equations")
        
        # Fit the model
        self.regressor.fit(X, y)
        
        # Extract results
        results = self._extract_results()
        
        # Update Pareto front
        self._update_pareto_front(results['equations'])
        
        return results
    
    def _extract_results(self) -> Dict[str, Any]:
        """Extract and format results from PySR."""
        equations = []
        
        # Get equation dataframe
        eq_df = self.regressor.equations_
        
        for idx, row in eq_df.iterrows():
            eq_info = {
                'sympy': row['sympy_format'],
                'complexity': row['complexity'],
                'loss': row['loss'],
                'score': row['score'],
                'equation_str': str(row['equation']),
                'generation': self.generation
            }
            equations.append(eq_info)
        
        # Best equation
        best_eq = self.regressor.sympy()
        best_complexity = self.regressor.get_best()['complexity']
        best_loss = self.regressor.get_best()['loss']
        
        results = {
            'best_equation': str(best_eq),
            'best_complexity': best_complexity,
            'best_loss': best_loss,
            'equations': equations,
            'pareto_front': self.pareto_front,
            'generation': self.generation
        }
        
        # Store best equation
        self.best_equations.append({
            'generation': self.generation,
            'equation': str(best_eq),
            'loss': best_loss,
            'complexity': best_complexity
        })
        
        logger.info(f"Best equation (complexity={best_complexity}, loss={best_loss:.6f}):")
        logger.info(f"  {best_eq}")
        
        return results
    
    def _update_pareto_front(self, equations: List[Dict]):
        """
        Update Pareto front of equations trading off accuracy vs complexity.
        """
        for eq in equations:
            # Check if this equation dominates any existing ones
            is_dominated = False
            to_remove = []
            
            for i, existing in enumerate(self.pareto_front):
                # Check domination
                if (eq['complexity'] <= existing['complexity'] and 
                    eq['loss'] < existing['loss']):
                    # New equation dominates existing one
                    to_remove.append(i)
                elif (eq['complexity'] >= existing['complexity'] and 
                      eq['loss'] > existing['loss']):
                    # Existing equation dominates new one
                    is_dominated = True
                    break
            
            # Remove dominated equations
            for idx in reversed(to_remove):
                self.pareto_front.pop(idx)
            
            # Add new equation if not dominated
            if not is_dominated:
                self.pareto_front.append(eq)
        
        # Sort by complexity
        self.pareto_front.sort(key=lambda x: x['complexity'])
        
        logger.info(f"Pareto front contains {len(self.pareto_front)} equations")
    
    def compile_to_function(self, equation_str: str, 
                           feature_names: List[str]) -> callable:
        """
        Compile a symbolic equation to a Python function.
        """
        try:
            # Parse with SymPy
            eq_sympy = sp.sympify(equation_str)
            
            # Create symbols for features
            symbols = [sp.Symbol(name) for name in feature_names]
            
            # Lambdify to create function
            func = sp.lambdify(symbols, eq_sympy, modules=['numpy', 'jax'])
            
            return func
        except Exception as e:
            logger.error(f"Failed to compile equation: {equation_str}")
            logger.error(f"Error: {e}")
            return lambda *args: 0.0
    
    def save_state(self, filepath: str):
        """Save the current state of the symbolic regression engine."""
        state = {
            'generation': self.generation,
            'best_equations': self.best_equations,
            'pareto_front': self.pareto_front,
            'config': asdict(self.config)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save PySR model separately
        if self.regressor and hasattr(self.regressor, 'equations_'):
            model_path = filepath.replace('.json', '_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.regressor, f)
        
        logger.info(f"Saved symbolic regression state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load a saved state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.generation = state['generation']
        self.best_equations = state['best_equations']
        self.pareto_front = state['pareto_front']
        
        # Load PySR model if available
        model_path = filepath.replace('.json', '_model.pkl')
        try:
            with open(model_path, 'rb') as f:
                self.regressor = pickle.load(f)
            logger.info(f"Loaded symbolic regression state from {filepath}")
        except FileNotFoundError:
            logger.warning("No saved PySR model found, using fresh initialization")


# =====================================================================
# NEURO-SYMBOLIC ORCHESTRATOR
# =====================================================================

class NeuroSymbolicOrchestrator:
    """
    Main orchestrator combining PINN training with symbolic discovery.
    Implements the full neuro-symbolic loop.
    """
    
    def __init__(self, pinn_trainer, symbolic_engine: SymbolicRegressionEngine,
                 simulation_engine):
        self.pinn_trainer = pinn_trainer
        self.symbolic_engine = symbolic_engine
        self.simulation = simulation_engine
        
        self.current_equation = None
        self.evolution_history = []
        self.convergence_threshold = 1e-4
        self.patience = 3
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def run_evolution_cycle(self, n_generations: int = 10):
        """
        Run the complete neuro-symbolic evolution cycle.
        """
        logger.info(f"Starting neuro-symbolic evolution for {n_generations} generations")
        
        for generation in range(n_generations):
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATION {generation + 1}/{n_generations}")
            logger.info(f"{'='*60}")
            
            # Step 1: Generate simulation data
            logger.info("Step 1: Running physics simulation...")
            sim_data = self.simulation.run_simulation(steps=1000, save_interval=10)
            
            # Step 2: Train PINN with current equation
            logger.info("Step 2: Training PINN...")
            
            # Prepare training data
            pinn_data = self._prepare_pinn_data(sim_data)
            
            # Train PINN
            self.pinn_trainer.train(pinn_data, epochs=1000)
            
            # Get residuals for symbolic regression
            residuals = self._compute_residuals(sim_data)
            
            # Step 3: Symbolic discovery
            logger.info("Step 3: Discovering symbolic equations...")
            
            # Prepare data for symbolic regression
            sr_data = self.symbolic_engine.prepare_physics_data(sim_data, residuals)
            
            # Extract features and target
            feature_cols = ['r', 'r_squared', 'r_inv', 'mass', 'momentum']
            X = sr_data[feature_cols]
            y = sr_data['target'].values
            
            # Discover equations
            results = self.symbolic_engine.discover_equation(X, y, feature_cols)
            
            # Step 4: Update physics
            logger.info("Step 4: Updating physics model...")
            
            # Compile best equation to function
            best_eq = results['best_equation']
            eq_func = self.symbolic_engine.compile_to_function(best_eq, feature_cols)
            
            # Update PINN with new physics
            self.pinn_trainer.set_symbolic_equation(eq_func)
            
            # Step 5: Evaluate convergence
            current_loss = results['best_loss']
            logger.info(f"Current best loss: {current_loss:.6f}")
            
            # Store evolution history
            self.evolution_history.append({
                'generation': generation + 1,
                'equation': best_eq,
                'loss': current_loss,
                'pinn_loss': self.pinn_trainer.loss_history[-1] if self.pinn_trainer.loss_history else None,
                'pareto_front_size': len(results['pareto_front'])
            })
            
            # Check convergence
            if self._check_convergence(current_loss):
                logger.info(f"Converged at generation {generation + 1}")
                break
            
            # Save checkpoint
            self._save_checkpoint(generation + 1)
        
        logger.info("Evolution complete!")
        return self._compile_results()
    
    def _prepare_pinn_data(self, sim_data: List[Dict]) -> Dict:
        """Prepare simulation data for PINN training."""
        import jax.numpy as jnp
        
        t_data = []
        x_data = []
        y_data = []
        z_data = []
        u_data = []
        
        for state in sim_data:
            for body in state['bodies']:
                t_data.append(state['time'])
                x_data.append(body['position'][0])
                y_data.append(body['position'][1])
                z_data.append(body['position'][2])
                # Use energy as target field
                u_data.append(state['energy']['total'])
        
        return {
            't_data': jnp.array(t_data),
            'x_data': jnp.array(x_data),
            'y_data': jnp.array(y_data),
            'z_data': jnp.array(z_data),
            'u_data': jnp.array(u_data)
        }
    
    def _compute_residuals(self, sim_data: List[Dict]) -> np.ndarray:
        """Compute physics residuals from PINN."""
        # Simplified - in production, compute actual PINN residuals
        residuals = []
        
        for i in range(len(sim_data) - 1):
            state = sim_data[i]
            next_state = sim_data[i + 1]
            
            # Compute time derivative approximation
            dt = next_state['time'] - state['time']
            energy_change = (next_state['energy']['total'] - 
                           state['energy']['total']) / dt
            
            # Residual is deviation from energy conservation
            residuals.append(abs(energy_change))
        
        return np.array(residuals)
    
    def _check_convergence(self, current_loss: float) -> bool:
        """Check if the evolution has converged."""
        if current_loss < self.best_loss - self.convergence_threshold:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience
    
    def _save_checkpoint(self, generation: int):
        """Save checkpoint of current state."""
        checkpoint = {
            'generation': generation,
            'evolution_history': self.evolution_history,
            'best_loss': self.best_loss
        }
        
        filepath = f"checkpoints/evolution_gen_{generation}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        # Save symbolic engine state
        self.symbolic_engine.save_state(f"checkpoints/symbolic_gen_{generation}.json")
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results of evolution."""
        return {
            'best_equation': self.symbolic_engine.best_equations[-1] if self.symbolic_engine.best_equations else None,
            'pareto_front': self.symbolic_engine.pareto_front,
            'evolution_history': self.evolution_history,
            'final_generation': len(self.evolution_history),
            'converged': self.patience_counter >= self.patience
        }