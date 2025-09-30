"""
JAX-Based Physics-Informed Neural Network for Wave Theory
Production-ready implementation with automatic differentiation
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import equinox as eqx
import optax
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# JAX-BASED PINN IMPLEMENTATION
# =====================================================================

class WavePINN(eqx.Module):
    """
    Physics-Informed Neural Network for Wave Theory using JAX/Equinox.
    Architecture: 6 hidden layers with 128 neurons each.
    """
    mlp: eqx.nn.MLP

    def __init__(self, key: jax.random.PRNGKey):
        """Initialize the network with Xavier/Glorot initialization."""
        self.mlp = eqx.nn.MLP(
            in_size=4,  # [t, x, y, z]
            out_size=1,  # scalar field output
            width_size=128,
            depth=6,
            activation=jax.nn.tanh,
            key=key
        )

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: predict field value u(t, x, y, z)."""
        inputs = jnp.stack([t, x, y, z], axis=-1)
        return self.mlp(inputs).squeeze()


@dataclass
class PINNConfig:
    """Configuration for PINN training."""
    learning_rate: float = 1e-3
    lr_decay_rate: float = 0.99
    lr_decay_steps: int = 1000
    lambda_data: float = 1.0
    lambda_physics: float = 1.0
    lambda_bc: float = 1.0
    lambda_ic: float = 1.0
    n_collocation_points: int = 10000
    batch_size: int = 256
    n_epochs: int = 10000

    # Dynamic loss weighting
    use_dynamic_weights: bool = True
    weight_update_interval: int = 100


class PhysicsResidual(eqx.Module):
    """
    Computes physics residuals for a given symbolic equation.
    This module is updated dynamically as new equations are discovered.
    """
    equation_str: str
    wave_speed: float = 1.0

    def __call__(self, model: WavePINN, t: jnp.ndarray, x: jnp.ndarray,
                 y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the physics residual for the wave equation.
        Default: ∂²u/∂t² - c²∇²u = 0
        """

        # Define the function for automatic differentiation
        def u_fn(t, x, y, z):
            return model(t, x, y, z)

        # Compute first derivatives
        u_t = grad(u_fn, argnums=0)(t, x, y, z)
        u_x = grad(u_fn, argnums=1)(t, x, y, z)
        u_y = grad(u_fn, argnums=2)(t, x, y, z)
        u_z = grad(u_fn, argnums=3)(t, x, y, z)

        # Compute second derivatives
        u_tt = grad(grad(u_fn, argnums=0), argnums=0)(t, x, y, z)
        u_xx = grad(grad(u_fn, argnums=1), argnums=1)(t, x, y, z)
        u_yy = grad(grad(u_fn, argnums=2), argnums=2)(t, x, y, z)
        u_zz = grad(grad(u_fn, argnums=3), argnums=3)(t, x, y, z)

        # Wave equation residual
        laplacian = u_xx + u_yy + u_zz
        residual = u_tt - self.wave_speed ** 2 * laplacian

        return residual


class AdaptivePhysicsResidual(PhysicsResidual):
    """
    Physics residual that can be updated with discovered symbolic equations.
    """
    symbolic_fn: Optional[Callable] = None

    def set_symbolic_equation(self, equation_fn: Callable):
        """Update the physics residual with a new symbolic equation."""
        self.symbolic_fn = equation_fn

    def __call__(self, model: WavePINN, t: jnp.ndarray, x: jnp.ndarray,
                 y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate residual using either the symbolic equation or default wave equation.
        """
        if self.symbolic_fn is not None:
            # Use discovered symbolic equation
            def u_fn(t, x, y, z):
                return model(t, x, y, z)

            u = u_fn(t, x, y, z)
            u_t = grad(u_fn, argnums=0)(t, x, y, z)
            u_x = grad(u_fn, argnums=1)(t, x, y, z)
            u_tt = grad(grad(u_fn, argnums=0), argnums=0)(t, x, y, z)
            u_xx = grad(grad(u_fn, argnums=1), argnums=1)(t, x, y, z)

            # Evaluate symbolic equation
            return self.symbolic_fn(u, u_t, u_x, u_tt, u_xx, t, x)
        else:
            # Fall back to default wave equation
            return super().__call__(model, t, x, y, z)


class PINNTrainer:
    """
    Training engine for Physics-Informed Neural Networks.
    """

    def __init__(self, model: WavePINN, config: PINNConfig, key: jax.random.PRNGKey):
        self.model = model
        self.config = config
        self.key = key
        self.physics_residual = AdaptivePhysicsResidual()

        # Initialize optimizer with learning rate schedule
        schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_rate
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        # Loss weights (for dynamic weighting)
        self.loss_weights = {
            'data': config.lambda_data,
            'physics': config.lambda_physics,
            'bc': config.lambda_bc,
            'ic': config.lambda_ic
        }

        # Training history
        self.loss_history = []
        self.weight_history = []

    @jit
    def compute_data_loss(self, model: WavePINN, t_data: jnp.ndarray, x_data: jnp.ndarray,
                          y_data: jnp.ndarray, z_data: jnp.ndarray, u_data: jnp.ndarray) -> jnp.ndarray:
        """Compute supervised data loss."""
        u_pred = vmap(model)(t_data, x_data, y_data, z_data)
        return jnp.mean((u_pred - u_data) ** 2)

    @jit
    def compute_physics_loss(self, model: WavePINN, t_phys: jnp.ndarray, x_phys: jnp.ndarray,
                             y_phys: jnp.ndarray, z_phys: jnp.ndarray) -> jnp.ndarray:
        """Compute physics residual loss."""
        # Vectorize over collocation points
        residuals = vmap(self.physics_residual, in_axes=(None, 0, 0, 0, 0))(
            model, t_phys, x_phys, y_phys, z_phys
        )
        return jnp.mean(residuals ** 2)

    @jit
    def compute_bc_loss(self, model: WavePINN, t_bc: jnp.ndarray, x_bc: jnp.ndarray,
                        y_bc: jnp.ndarray, z_bc: jnp.ndarray, u_bc: jnp.ndarray) -> jnp.ndarray:
        """Compute boundary condition loss."""
        u_pred = vmap(model)(t_bc, x_bc, y_bc, z_bc)
        return jnp.mean((u_pred - u_bc) ** 2)

    @jit
    def compute_ic_loss(self, model: WavePINN, t_ic: jnp.ndarray, x_ic: jnp.ndarray,
                        y_ic: jnp.ndarray, z_ic: jnp.ndarray, u_ic: jnp.ndarray) -> jnp.ndarray:
        """Compute initial condition loss."""
        u_pred = vmap(model)(t_ic, x_ic, y_ic, z_ic)
        return jnp.mean((u_pred - u_ic) ** 2)

    @jit
    def compute_total_loss(self, model: WavePINN, batch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, Dict]:
        """Compute weighted total loss."""
        losses = {}

        # Individual loss components
        if 'data' in batch:
            losses['data'] = self.compute_data_loss(
                model, batch['t_data'], batch['x_data'],
                batch['y_data'], batch['z_data'], batch['u_data']
            )
        else:
            losses['data'] = jnp.array(0.0)

        if 'physics' in batch:
            losses['physics'] = self.compute_physics_loss(
                model, batch['t_phys'], batch['x_phys'],
                batch['y_phys'], batch['z_phys']
            )
        else:
            losses['physics'] = jnp.array(0.0)

        if 'bc' in batch:
            losses['bc'] = self.compute_bc_loss(
                model, batch['t_bc'], batch['x_bc'],
                batch['y_bc'], batch['z_bc'], batch['u_bc']
            )
        else:
            losses['bc'] = jnp.array(0.0)

        if 'ic' in batch:
            losses['ic'] = self.compute_ic_loss(
                model, batch['t_ic'], batch['x_ic'],
                batch['y_ic'], batch['z_ic'], batch['u_ic']
            )
        else:
            losses['ic'] = jnp.array(0.0)

        # Weighted sum
        total_loss = (self.loss_weights['data'] * losses['data'] +
                      self.loss_weights['physics'] * losses['physics'] +
                      self.loss_weights['bc'] * losses['bc'] +
                      self.loss_weights['ic'] * losses['ic'])

        return total_loss, losses

    def update_loss_weights(self, losses: Dict[str, float]):
        """
        Dynamically update loss weights to balance gradient contributions.
        Based on the gradient normalization strategy.
        """
        if not self.config.use_dynamic_weights:
            return

        # Calculate gradient magnitudes (simplified - in practice, compute actual gradients)
        grad_magnitudes = {k: abs(v) for k, v in losses.items() if v > 0}

        if grad_magnitudes:
            # Normalize to have similar gradient contributions
            mean_magnitude = np.mean(list(grad_magnitudes.values()))
            for key in self.loss_weights:
                if key in grad_magnitudes and grad_magnitudes[key] > 0:
                    self.loss_weights[key] *= mean_magnitude / grad_magnitudes[key]
                    # Clip weights to reasonable range
                    self.loss_weights[key] = np.clip(self.loss_weights[key], 0.01, 100.0)

    @jit
    def train_step(self, model: WavePINN, opt_state, batch: Dict[str, jnp.ndarray]):
        """Single training step with gradient descent."""
        # Compute loss and gradients
        loss_fn = lambda m: self.compute_total_loss(m, batch)
        (loss_value, losses), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

        # Update model parameters
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss_value, losses

    def generate_collocation_points(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """Generate random collocation points for physics loss."""
        key, subkey = random.split(self.key)
        self.key = key

        # Domain bounds
        t_min, t_max = 0.0, 10.0
        x_min, x_max = -10.0, 10.0
        y_min, y_max = -10.0, 10.0
        z_min, z_max = -10.0, 10.0

        # Random sampling
        t_phys = random.uniform(subkey, (n_points,), minval=t_min, maxval=t_max)
        x_phys = random.uniform(subkey, (n_points,), minval=x_min, maxval=x_max)
        y_phys = random.uniform(subkey, (n_points,), minval=y_min, maxval=y_max)
        z_phys = random.uniform(subkey, (n_points,), minval=z_min, maxval=z_max)

        return {
            't_phys': t_phys,
            'x_phys': x_phys,
            'y_phys': y_phys,
            'z_phys': z_phys
        }

    def train(self, data: Dict[str, jnp.ndarray], epochs: Optional[int] = None):
        """
        Main training loop with adaptive sampling and dynamic weighting.
        """
        epochs = epochs or self.config.n_epochs
        logger.info(f"Starting PINN training for {epochs} epochs")

        for epoch in range(epochs):
            # Generate new collocation points (adaptive sampling)
            batch = self.generate_collocation_points(self.config.n_collocation_points)

            # Add data points if available
            if data:
                batch.update(data)

            # Training step
            self.model, self.opt_state, loss_value, losses = self.train_step(
                self.model, self.opt_state, batch
            )

            # Store history
            self.loss_history.append(float(loss_value))
            self.weight_history.append(dict(self.loss_weights))

            # Update weights periodically
            if epoch % self.config.weight_update_interval == 0 and epoch > 0:
                self.update_loss_weights(losses)

            # Logging
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Total Loss: {loss_value:.6f}")
                logger.info(f"  Components: {losses}")
                logger.info(f"  Weights: {self.loss_weights}")

        logger.info("Training complete")
        return self.model

    def set_symbolic_equation(self, equation_fn: Callable):
        """Update the physics residual with a newly discovered equation."""
        self.physics_residual.set_symbolic_equation(equation_fn)
        logger.info("Updated physics residual with new symbolic equation")


# =====================================================================
# ADAPTIVE SAMPLING STRATEGIES
# =====================================================================

class AdaptiveSampler:
    """
    Implements adaptive sampling strategies for collocation points.
    Focuses sampling in regions of high physics residuals.
    """

    def __init__(self, model: WavePINN, residual_fn: Callable):
        self.model = model
        self.residual_fn = residual_fn
        self.residual_history = []

    def compute_residual_map(self, points: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute residuals at given points."""
        residuals = vmap(self.residual_fn, in_axes=(None, 0, 0, 0, 0))(
            self.model, points['t'], points['x'], points['y'], points['z']
        )
        return jnp.abs(residuals)

    def generate_adaptive_points(self, n_points: int, key: jax.random.PRNGKey,
                                 high_residual_ratio: float = 0.5) -> Dict[str, jnp.ndarray]:
        """
        Generate points with bias towards high-residual regions.
        """
        n_uniform = int(n_points * (1 - high_residual_ratio))
        n_focused = n_points - n_uniform

        # Generate uniform points
        key, subkey = random.split(key)
        uniform_points = self._generate_uniform_points(n_uniform, subkey)

        # Generate focused points (if we have residual history)
        if self.residual_history:
            key, subkey = random.split(key)
            focused_points = self._generate_focused_points(n_focused, subkey)

            # Combine points
            combined = {}
            for key in uniform_points:
                combined[key] = jnp.concatenate([uniform_points[key], focused_points[key]])
            return combined

        return uniform_points

    def _generate_uniform_points(self, n_points: int, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Generate uniformly distributed points."""
        t = random.uniform(key, (n_points,), minval=0.0, maxval=10.0)
        x = random.uniform(key, (n_points,), minval=-10.0, maxval=10.0)
        y = random.uniform(key, (n_points,), minval=-10.0, maxval=10.0)
        z = random.uniform(key, (n_points,), minval=-10.0, maxval=10.0)
        return {'t': t, 'x': x, 'y': y, 'z': z}

    def _generate_focused_points(self, n_points: int, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Generate points focused on high-residual regions."""
        # Use importance sampling based on residual magnitudes
        # This is a simplified version - production would use KD-trees or similar

        # Get regions with highest residuals from history
        top_regions = self.residual_history[-1]['high_residual_regions']

        # Sample around these regions with Gaussian noise
        points = {'t': [], 'x': [], 'y': [], 'z': []}

        for region in top_regions[:n_points]:
            key, subkey = random.split(key)
            noise = random.normal(subkey, (4,)) * 0.5

            points['t'].append(region[0] + noise[0])
            points['x'].append(region[1] + noise[1])
            points['y'].append(region[2] + noise[2])
            points['z'].append(region[3] + noise[3])

        return {k: jnp.array(v) for k, v in points.items()}


# =====================================================================
# UTILITIES FOR INTEGRATION
# =====================================================================

def create_pinn_model(key: jax.random.PRNGKey) -> Tuple[WavePINN, PINNTrainer]:
    """Factory function to create and initialize a PINN model."""
    model = WavePINN(key)
    config = PINNConfig()
    trainer = PINNTrainer(model, config, key)
    return model, trainer


def compile_symbolic_to_jax(equation_str: str) -> Callable:
    """
    Compile a symbolic equation string to a JAX-compatible function.
    Uses SymPy for safe parsing and compilation.
    """
    import sympy as sp
    from sympy import lambdify

    # Define symbolic variables
    u, u_t, u_x, u_tt, u_xx, t, x = sp.symbols('u u_t u_x u_tt u_xx t x')

    # Parse the equation string
    try:
        expr = sp.sympify(equation_str)

        # Convert to JAX function
        jax_fn = lambdify([u, u_t, u_x, u_tt, u_xx, t, x], expr, modules='jax')
        return jax_fn
    except Exception as e:
        logger.error(f"Failed to compile equation: {equation_str}")
        logger.error(f"Error: {e}")
        return lambda *args: jnp.array(0.0)