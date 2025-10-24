"""
JAX-Based Physics-Informed Neural Network for Wave Theory
Production-ready implementation with automatic differentiation
Includes PsiNetwork modulation based on psychotronic/holographic principles.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, jacfwd, value_and_grad
import equinox as eqx
import optax
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging
import sympy as sp
from sympy import lambdify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# Configuration
# =====================================================================

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
    batch_size: int = 256 # Note: Batching needs explicit implementation in train loop if desired
    n_epochs: int = 10000
    base_wave_speed: float = 1.0 # Added base wave speed

    # Dynamic loss weighting
    use_dynamic_weights: bool = True
    weight_update_interval: int = 100

    # Domain bounds for collocation point generation (moved here for clarity)
    t_min: float = 0.0
    t_max: float = 1.0
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = -1.0
    z_max: float = 1.0


# =====================================================================
# Neural Network Models (Equinox Modules)
# =====================================================================

class WavePINN(eqx.Module):
    """
    Neural Network predicting the wave field u(t, x, y, z).
    Architecture: MLP with configurable depth and width.
    """
    mlp: eqx.nn.MLP

    def __init__(self, key: jax.random.PRNGKey, in_size=4, out_size=1, width_size=128, depth=6):
        """Initialize the network with Xavier/Glorot initialization."""
        self.mlp = eqx.nn.MLP(
            in_size=in_size, # [t, x, y, z] or fewer dimensions
            out_size=out_size, # scalar field output u
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key
        )

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Forward pass using channels-first (features-first) convention expected by Equinox Linear.
        Accepts scalars or 1D arrays; supports vmap over scalars.
        """
        # Stack into shape (features, ...) where features=4 and optional trailing dims are batch-like
        inputs = jnp.stack([t, x, y, z], axis=0)  # shape: (4,) or (4, N)

        # Pass through MLP; Equinox Linear uses weight @ x, expecting features on axis 0
        output = self.mlp(inputs)  # shape: (out_size,) or (out_size, N)

        # If out_size==1, squeeze the feature axis
        if output.shape[0] == 1:
            output = output[0]  # shape: () or (N,)
        return output


class PsiNetwork(eqx.Module):
    """
    Neural Network representing the modulating consciousness/holographic field.
    Outputs parameters that influence the physics (e.g., local wave speed).
    """
    mlp: eqx.nn.MLP
    output_dim: int # Number of modulation parameters to output

    def __init__(self, key: jax.random.PRNGKey, in_size=4, output_dim=1, width_size=32, depth=2):
        """Initialize a smaller MLP for the psi field."""
        self.output_dim = output_dim
        self.mlp = eqx.nn.MLP(
            in_size=in_size, # [t, x, y, z]
            out_size=output_dim, # Modulation parameter(s)
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh, # Tanh often good for outputs bounded around 0
            key=key
        )

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """Forward pass using channels-first convention consistent with Equinox Linear."""
        inputs = jnp.stack([t, x, y, z], axis=0)  # shape: (4,) or (4, N)
        output = self.mlp(inputs)  # shape: (output_dim,) or (output_dim, N)
        if self.output_dim == 1:
            output = output[0]  # shape: () or (N,)
        return output


# =====================================================================
# Physics Residual Calculation
# =====================================================================

class PhysicsResidual(eqx.Module):
    """
    Computes physics residuals, incorporating PsiNetwork modulation.
    Can be updated dynamically with symbolic equations.
    """
    base_wave_speed: float = eqx.static_field()
    symbolic_fn: Optional[Callable] = eqx.static_field(default=None)
    symbolic_vars: Optional[tuple] = eqx.static_field(default=None) # Store symbols needed by symbolic_fn

    def __init__(self, base_wave_speed: float = 1.0):
       self.base_wave_speed = base_wave_speed

    def set_symbolic_equation(self, equation_fn: Callable, variables: tuple):
        """Update the physics residual with a new symbolic equation and its variables."""
        # Use object.__setattr__ because the class is frozen by eqx.Module
        object.__setattr__(self, 'symbolic_fn', equation_fn)
        object.__setattr__(self, 'symbolic_vars', variables)
        logger.info(f"Set symbolic equation requiring variables: {variables}")


    # Define the core function for evaluating u and its derivatives
    @staticmethod
    # @eqx.filter_jit # Jitting static methods can be tricky, test performance impact
    def _evaluate_u_and_derivatives(u_model: WavePINN, psi_model: PsiNetwork,
                                    t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray,
                                    base_wave_speed: float) -> dict:
        """Calculates u and its necessary derivatives using JAX's autodiff for SINGLE point."""

        # Ensure inputs are scalar for grad functions wrt coordinates
        t_scalar, x_scalar, y_scalar, z_scalar = map(jnp.squeeze, [t, x, y, z])

        # Wrap model call for grad wrt coordinates.
        def u_fn_coords(_t, _x, _y, _z):
             # Call the model, ensuring it handles scalar inputs correctly
            return u_model(jnp.array(_t), jnp.array(_x), jnp.array(_y), jnp.array(_z))

        # Calculate psi modulation first (ensure psi_model handles scalar inputs)
        psi_output = psi_model(jnp.array(t_scalar), jnp.array(x_scalar), jnp.array(y_scalar), jnp.array(z_scalar))
        modulated_c = base_wave_speed * (1 + psi_output) # Assumes psi_output is scalar

        # Get value of u
        u = u_fn_coords(t_scalar, x_scalar, y_scalar, z_scalar)

        # --- Calculate Derivatives ---
        # First derivatives
        u_t = grad(u_fn_coords, argnums=0)(t_scalar, x_scalar, y_scalar, z_scalar)
        u_x = grad(u_fn_coords, argnums=1)(t_scalar, x_scalar, y_scalar, z_scalar)
        u_y = grad(u_fn_coords, argnums=2)(t_scalar, x_scalar, y_scalar, z_scalar)
        u_z = grad(u_fn_coords, argnums=3)(t_scalar, x_scalar, y_scalar, z_scalar)

        # Second derivatives (Hessian diagonals essentially)
        # Need to compute gradient of the gradient function
        u_tt = grad(lambda t_: grad(u_fn_coords, argnums=0)(t_, x_scalar, y_scalar, z_scalar))(t_scalar)
        u_xx = grad(lambda x_: grad(u_fn_coords, argnums=1)(t_scalar, x_, y_scalar, z_scalar))(x_scalar)
        u_yy = grad(lambda y_: grad(u_fn_coords, argnums=2)(t_scalar, x_scalar, y_, z_scalar))(y_scalar)
        u_zz = grad(lambda z_: grad(u_fn_coords, argnums=3)(t_scalar, x_scalar, y_scalar, z_))(z_scalar)
        # --- End Derivatives ---

        return {
            "u": u, "u_t": u_t, "u_x": u_x, "u_y": u_y, "u_z": u_z,
            "u_tt": u_tt, "u_xx": u_xx, "u_yy": u_yy, "u_zz": u_zz,
            "modulated_c": modulated_c, "psi": psi_output,
            "t": t_scalar, "x": x_scalar, "y": y_scalar, "z": z_scalar # Pass coordinates too
        }

    # This __call__ expects scalar t, x, y, z inputs
    def __call__(self, u_model: WavePINN, psi_model: PsiNetwork,
                 t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the physics residual based on current configuration for a single point.
        Handles both default modulated wave equation and symbolic equations.
        """
        # Get u and its derivatives, plus the modulated wave speed
        derivs = self._evaluate_u_and_derivatives(u_model, psi_model, t, x, y, z, self.base_wave_speed)

        if self.symbolic_fn is not None and self.symbolic_vars is not None:
            # Prepare arguments for the symbolic function based on self.symbolic_vars
            args_for_symbolic = []
            for var_symbol in self.symbolic_vars:
                var_name = var_symbol.name
                if var_name in derivs:
                    args_for_symbolic.append(derivs[var_name])
                else:
                    # Handle case where a variable needed by the symbolic fn wasn't calculated
                    logger.error(f"Symbolic function needs variable '{var_name}' but it was not found in derivatives dict.")
                    return jnp.array(jnp.nan) # Return NaN or raise error

            # Evaluate discovered symbolic equation
            try:
                residual = self.symbolic_fn(*args_for_symbolic)
            except Exception as e:
                logger.error(f"Error evaluating symbolic function: {e}")
                logger.error(f"Args provided: {args_for_symbolic}")
                residual = jnp.array(jnp.nan)

        else:
            # Default: Modulated wave equation residual: u_tt - c(psi)² * ∇²u = 0
            # Handle potential NaN in derivatives using JAX-friendly control flow
            nan_flags = jnp.stack([jnp.any(jnp.isnan(derivs[k])) for k in ["u_tt", "u_xx", "u_yy", "u_zz", "modulated_c"]])
            has_nan = jnp.any(nan_flags)

            laplacian = derivs["u_xx"] + derivs["u_yy"] + derivs["u_zz"]
            resid_value = derivs["u_tt"] - derivs["modulated_c"]**2 * laplacian
            residual = jnp.where(has_nan, jnp.array(jnp.nan, dtype=jnp.float32), resid_value)

        # Ensure residual is a scalar float; avoid Python-side NaN checks inside JIT/vmap contexts
        residual_out = jnp.asarray(residual, dtype=jnp.float32)
        return residual_out


# =====================================================================
# PINN Trainer
# =====================================================================

class PINNTrainer:
    """
    Training engine for Physics-Informed Neural Networks.
    Manages model parameters, optimizer state, and training loop.
    Includes dynamic loss weighting.
    """
    model: WavePINN # The main network predicting 'u'
    psi_model: PsiNetwork # The network predicting modulation 'psi'
    config: PINNConfig
    key: jax.random.PRNGKey
    physics_residual: PhysicsResidual
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState

    # Loss weights (mutable dictionary for dynamic updates)
    loss_weights: Dict[str, float]

    # Training history (standard Python lists)
    loss_history: list
    weight_history: list

    def __init__(self, model: WavePINN, psi_model: PsiNetwork, config: PINNConfig, key: jax.random.PRNGKey):
        self.model = model
        self.psi_model = psi_model
        self.config = config
        self.key = key
        # Pass base_wave_speed from config to PhysicsResidual instance
        self.physics_residual = PhysicsResidual(base_wave_speed=config.base_wave_speed)

        # Log JAX backend/devices for transparency
        try:
            backend = jax.default_backend()
            devices = jax.devices()
            logger.info(f"JAX backend: {backend}; devices: {[d.platform + ':' + d.device_kind for d in devices]}")
        except Exception:
            pass

        # Initialize optimizer with learning rate schedule
        schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_rate
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        # Combine parameters from both models for the optimizer
        trainable_model_params = eqx.filter(self.model, eqx.is_array)
        trainable_psi_params = eqx.filter(self.psi_model, eqx.is_array)
        self.combined_params = {"u_params": trainable_model_params, "psi_params": trainable_psi_params}
        self.opt_state = self.optimizer.init(self.combined_params)

        # Loss weights (mutable dictionary stored directly)
        self.loss_weights = {
            'data': config.lambda_data,
            'physics': config.lambda_physics,
            'bc': config.lambda_bc,
            'ic': config.lambda_ic
        }

        # Training history (standard Python lists)
        self.loss_history = []
        self.weight_history = []


    # --- Loss Computation Functions (JIT compiled) ---
    # These take combined_params and extract the relevant part for the model

    @eqx.filter_jit
    def compute_data_loss(self, combined_params: Dict, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute supervised data loss."""
        # Rebuild the model temporarily with current params for this step
        current_u_params = combined_params["u_params"]
        static_model = eqx.filter(self.model, eqx.is_array, inverse=True)
        model = eqx.combine(current_u_params, static_model)

        # Vectorize the model call over the batch dimension
        u_pred = vmap(model, in_axes=(0, 0, 0, 0))(batch['t_data'], batch['x_data'], batch['y_data'], batch['z_data'])
        return jnp.mean((u_pred - batch['u_data']) ** 2)

    @eqx.filter_jit
    def compute_physics_loss(self, combined_params: Dict, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute physics residual loss."""
        # Rebuild models temporarily with current params
        current_u_params = combined_params["u_params"]
        static_model = eqx.filter(self.model, eqx.is_array, inverse=True)
        model = eqx.combine(current_u_params, static_model)

        current_psi_params = combined_params["psi_params"]
        static_psi_model = eqx.filter(self.psi_model, eqx.is_array, inverse=True)
        psi_model = eqx.combine(current_psi_params, static_psi_model)

        # Vectorize the residual calculation over collocation points
        residuals = vmap(self.physics_residual, in_axes=(None, None, 0, 0, 0, 0))(
            model, psi_model, batch['t_phys'], batch['x_phys'], batch['y_phys'], batch['z_phys']
        )
        # Add safeguard for NaNs
        safe_residuals = jnp.where(jnp.isnan(residuals), 0.0, residuals)
        return jnp.mean(safe_residuals ** 2)


    @eqx.filter_jit
    def compute_bc_loss(self, combined_params: Dict, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute boundary condition loss."""
        current_u_params = combined_params["u_params"]
        static_model = eqx.filter(self.model, eqx.is_array, inverse=True)
        model = eqx.combine(current_u_params, static_model)

        u_pred = vmap(model, in_axes=(0, 0, 0, 0))(batch['t_bc'], batch['x_bc'], batch['y_bc'], batch['z_bc'])
        return jnp.mean((u_pred - batch['u_bc']) ** 2)

    @eqx.filter_jit
    def compute_ic_loss(self, combined_params: Dict, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute initial condition loss."""
        current_u_params = combined_params["u_params"]
        static_model = eqx.filter(self.model, eqx.is_array, inverse=True)
        model = eqx.combine(current_u_params, static_model)

        u_pred = vmap(model, in_axes=(0, 0, 0, 0))(batch['t_ic'], batch['x_ic'], batch['y_ic'], batch['z_ic'])
        return jnp.mean((u_pred - batch['u_ic']) ** 2)

    # --- Total Loss and Training Step (JIT compiled) ---

    def _compute_total_loss_internal(self, combined_params: Dict, batch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, Dict]:
        """Helper to compute loss components and weighted sum."""
        losses = {}
        zero_loss_fn = lambda cp, b: jnp.array(0.0) # Ensure function signature matches

        losses['data'] = jax.lax.cond('t_data' in batch and batch['t_data'].size > 0,
                                     self.compute_data_loss, zero_loss_fn,
                                     combined_params, batch)
        losses['physics'] = jax.lax.cond('t_phys' in batch and batch['t_phys'].size > 0,
                                        self.compute_physics_loss, zero_loss_fn,
                                        combined_params, batch)
        losses['bc'] = jax.lax.cond('t_bc' in batch and batch['t_bc'].size > 0,
                                    self.compute_bc_loss, zero_loss_fn,
                                    combined_params, batch)
        losses['ic'] = jax.lax.cond('t_ic' in batch and batch['t_ic'].size > 0,
                                    self.compute_ic_loss, zero_loss_fn,
                                    combined_params, batch)

        # Weighted sum: Access weights from self.
        total_loss = (self.loss_weights['data'] * losses['data'] +
                      self.loss_weights['physics'] * losses['physics'] +
                      self.loss_weights['bc'] * losses['bc'] +
                      self.loss_weights['ic'] * losses['ic'])

        # Safeguard against NaN total loss if components are NaN
        total_loss = jnp.where(jnp.isnan(total_loss), jnp.inf, total_loss)

        return total_loss, losses

    @eqx.filter_jit
    def train_step(self, combined_params: Dict, opt_state, batch: Dict[str, jnp.ndarray]):
        """Single training step with gradient descent."""
        # Use value_and_grad with the internal loss function
        (loss_value, losses), grads = value_and_grad(self._compute_total_loss_internal, has_aux=True)(combined_params, batch)

        # Add gradient clipping to prevent explosion
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

        # Update model parameters
        updates, opt_state = self.optimizer.update(grads, opt_state, combined_params)
        combined_params = eqx.apply_updates(combined_params, updates)

        return combined_params, opt_state, loss_value, losses


    # --- Dynamic Weighting (Executed outside JIT) ---
    def update_loss_weights(self, losses: Dict[str, jnp.ndarray]):
        """
        Dynamically update loss weights based on loss magnitudes (proxy for gradients).
        """
        if not self.config.use_dynamic_weights:
            return

        # Ensure values are non-NaN before processing
        losses_float = {k: float(v) for k, v in losses.items() if not np.isnan(float(v))}
        loss_magnitudes = {k: abs(v) for k, v in losses_float.items() if v > 1e-12} # Avoid zero losses

        if not loss_magnitudes:
             return

        active_weights = {k: self.loss_weights[k] for k in loss_magnitudes}
        contributions = {k: active_weights[k] * loss_magnitudes[k] for k in active_weights}

        mean_contribution = np.mean(list(contributions.values()))

        if mean_contribution < 1e-12:
             return

        for key in self.loss_weights:
            if key in loss_magnitudes:
                # Add epsilon to prevent division by zero in rare cases
                target_weight = mean_contribution / (loss_magnitudes[key] + 1e-12)

                ema_alpha = 0.1
                current_weight = self.loss_weights[key]
                new_weight = (1 - ema_alpha) * current_weight + ema_alpha * target_weight

                self.loss_weights[key] = float(np.clip(new_weight, 0.01, 100.0))


    # --- Data Generation (Executed outside JIT) ---
    def generate_collocation_points(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """Generate random collocation points for physics loss using config bounds."""
        key, subkey = random.split(self.key)
        self.key = key # Update the key state

        t_phys = random.uniform(subkey, (n_points, 1), minval=self.config.t_min, maxval=self.config.t_max)
        key, subkey = random.split(key)
        x_phys = random.uniform(subkey, (n_points, 1), minval=self.config.x_min, maxval=self.config.x_max)
        key, subkey = random.split(key)
        y_phys = random.uniform(subkey, (n_points, 1), minval=self.config.y_min, maxval=self.config.y_max)
        key, subkey = random.split(key)
        z_phys = random.uniform(subkey, (n_points, 1), minval=self.config.z_min, maxval=self.config.z_max)

        # Ensure shapes are (N, 1) for consistency if needed later, but vmap usually handles (N,)
        return {
            't_phys': t_phys.squeeze(-1),
            'x_phys': x_phys.squeeze(-1),
            'y_phys': y_phys.squeeze(-1),
            'z_phys': z_phys.squeeze(-1)
        }


    # --- Main Training Loop (Executed outside JIT) ---
    def train(self, data: Dict[str, jnp.ndarray], epochs: Optional[int] = None):
        """
        Main training loop. 'data' contains supervised/BC/IC points.
        """
        epochs = epochs or self.config.n_epochs
        logger.info(f"Starting PINN training for {epochs} epochs")

        # Ensure initial parameters are finite
        is_finite_tree = lambda tree: jax.tree_util.tree_all(
            jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), eqx.filter(tree, eqx.is_array))
        )
        if not is_finite_tree(self.combined_params):
             logger.error("Initial parameters contain NaN or Inf!")
             return None, None # Or raise error

        for epoch in range(epochs):
            # 1. Generate collocation points
            physics_batch = self.generate_collocation_points(self.config.n_collocation_points)

            # 2. Prepare full batch for the training step
            full_batch = {
                't_data': data.get('t_data', jnp.array([])), 'x_data': data.get('x_data', jnp.array([])),
                'y_data': data.get('y_data', jnp.array([])), 'z_data': data.get('z_data', jnp.array([])),
                'u_data': data.get('u_data', jnp.array([])),

                't_phys': physics_batch['t_phys'], 'x_phys': physics_batch['x_phys'],
                'y_phys': physics_batch['y_phys'], 'z_phys': physics_batch['z_phys'],

                't_bc': data.get('t_bc', jnp.array([])), 'x_bc': data.get('x_bc', jnp.array([])),
                'y_bc': data.get('y_bc', jnp.array([])), 'z_bc': data.get('z_bc', jnp.array([])),
                'u_bc': data.get('u_bc', jnp.array([])),

                't_ic': data.get('t_ic', jnp.array([])), 'x_ic': data.get('x_ic', jnp.array([])),
                'y_ic': data.get('y_ic', jnp.array([])), 'z_ic': data.get('z_ic', jnp.array([])),
                'u_ic': data.get('u_ic', jnp.array([]))
            }
            batch_for_step = full_batch

            # 3. Perform training step
            try:
                # Check for NaNs/Infs BEFORE the step
                if not is_finite_tree(self.combined_params):
                    logger.error(f"NaN/Inf detected in parameters BEFORE step {epoch}!")
                    break
                if not is_finite_tree(self.opt_state):  # Check optimizer state too (arrays only)
                    logger.error(f"NaN/Inf detected in optimizer state BEFORE step {epoch}!")
                    break

                self.combined_params, self.opt_state, loss_value, losses = self.train_step(
                    self.combined_params, self.opt_state, batch_for_step
                )

                # Check for NaNs/Infs AFTER the step
                if not is_finite_tree(self.combined_params):
                    logger.error(f"NaN/Inf detected in parameters AFTER step {epoch}!")
                    break
                if not is_finite_tree(self.opt_state):
                    logger.error(f"NaN/Inf detected in optimizer state AFTER step {epoch}!")
                    break
                if jnp.isnan(loss_value) or jnp.isinf(loss_value):
                     logger.error(f"NaN/Inf loss ({loss_value}) detected at epoch {epoch}. Stopping training.")
                     break # Stop training if loss explodes


            except Exception as e:
                 logger.error(f"Error during training step at epoch {epoch}: {e}")
                 # You might want to log the batch content here for debugging
                 # logger.error(f"Batch keys: {batch_for_step.keys()}")
                 # logger.error(f"Batch shapes: { {k: v.shape for k, v in batch_for_step.items()} }")
                 raise e

            # 4. Store history
            self.loss_history.append(float(loss_value))
            self.weight_history.append(dict(self.loss_weights)) # Store copy

            # 5. Update weights periodically
            if self.config.use_dynamic_weights and epoch % self.config.weight_update_interval == 0 and epoch > 0:
                self.update_loss_weights(losses) # Pass JAX arrays

            # 6. Logging
            if epoch % 100 == 0 or epoch == epochs - 1:
                losses_py = {k: float(v) for k, v in losses.items()}
                logger.info(f"Epoch {epoch}/{epochs} - Total Loss: {float(loss_value):.6f}")
                log_losses = {k: f"{v:.4e}" for k,v in losses_py.items() if abs(v)>1e-12 and not np.isnan(v)}
                logger.info(f"  Components: {log_losses}")
                log_weights = {k: f"{v:.2f}" for k,v in self.loss_weights.items()}
                logger.info(f"  Weights: {log_weights}")

        # Final update to model instances
        final_u_params = self.combined_params["u_params"]
        static_model = eqx.filter(self.model, eqx.is_array, inverse=True)
        self.model = eqx.combine(final_u_params, static_model)

        final_psi_params = self.combined_params["psi_params"]
        static_psi_model = eqx.filter(self.psi_model, eqx.is_array, inverse=True)
        self.psi_model = eqx.combine(final_psi_params, static_psi_model)


        logger.info("Training complete or stopped. Models updated with final parameters.")
        return self.model, self.psi_model # Return both trained models

    def set_symbolic_equation(self, equation_fn: Callable, variables: tuple):
        """Update the physics residual function."""
        self.physics_residual.set_symbolic_equation(equation_fn, variables)
        logger.info(f"Updated physics residual with new symbolic equation using vars: {variables}")

# =====================================================================
# Utilities
# =====================================================================

def create_pinn_system(key: jax.random.PRNGKey, config: PINNConfig) -> Tuple[WavePINN, PsiNetwork, PINNTrainer]:
    """Factory function to create and initialize the full PINN system."""
    key, model_key, psi_key, trainer_key = random.split(key, 4) # Need separate key for trainer
    model = WavePINN(model_key)
    psi_model = PsiNetwork(psi_key) # Default psi network
    trainer = PINNTrainer(model, psi_model, config, trainer_key) # Pass distinct key
    return model, psi_model, trainer


def create_pinn_model(key: jax.random.PRNGKey, config: Optional[PINNConfig] = None) -> Tuple[WavePINN, PINNTrainer]:
    """Backward-compatible factory: returns (model, trainer).
    Note: A PsiNetwork is created internally and managed by the trainer.
    """
    config = config or PINNConfig()
    model, _psi, trainer = create_pinn_system(key, config)
    return model, trainer

def compile_symbolic_to_jax(equation_str: str) -> Tuple[Optional[Callable], Optional[tuple]]:
    """
    Compile a symbolic equation string to a JAX-compatible function.
    Uses SymPy for safe parsing and compilation. Returns function and required variables.
    """
    # Define symbolic variables likely to be used
    u, u_t, u_x, u_y, u_z = sp.symbols('u u_t u_x u_y u_z')
    u_tt, u_xx, u_yy, u_zz = sp.symbols('u_tt u_xx u_yy u_zz')
    t, x, y, z = sp.symbols('t x y z')
    psi, modulated_c = sp.symbols('psi modulated_c') # Variables from our system

    available_symbols = {
        'u': u, 'u_t': u_t, 'u_x': u_x, 'u_y': u_y, 'u_z': u_z,
        'u_tt': u_tt, 'u_xx': u_xx, 'u_yy': u_yy, 'u_zz': u_zz,
        't': t, 'x': x, 'y': y, 'z': z,
        'psi': psi, 'modulated_c': modulated_c,
        'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'pi': sp.pi, # Common functions
        'tanh': sp.tanh, 'sqrt': sp.sqrt
    }

    try:
        # Safely evaluate the expression string using the defined symbols
        expr = sp.sympify(equation_str, locals=available_symbols)

        # Identify free symbols actually used in the expression
        free_vars = tuple(sorted(expr.free_symbols, key=lambda s: s.name))

        # Check if all free_vars are in our known 'available_symbols' names
        unknown_vars = [v for v in free_vars if v.name not in available_symbols]
        if unknown_vars:
            logger.error(f"Equation '{equation_str}' contains unknown symbols: {unknown_vars}")
            return None, None

        # Convert to JAX function using jax.numpy
        jax_fn = lambdify(free_vars, expr, modules=['jax.numpy', 'jax'])

        logger.info(f"Successfully compiled: {equation_str} -> requires args in order: {free_vars}")
        return jax_fn, free_vars
    except (sp.SympifyError, TypeError, NameError, SyntaxError) as e:
        logger.error(f"Failed to compile equation: {equation_str}")
        logger.error(f"Error type: {type(e).__name__}, Message: {e}")
        return None, None


# =====================================================================
# Example Usage (Corrected)
# =====================================================================
if __name__ == "__main__":
    # --- Initialization ---
    key = random.PRNGKey(42)
    # Define domain bounds directly for the example
    t_min_ex, t_max_ex = 0.0, 1.0
    x_min_ex, x_max_ex = -1.0, 1.0
    y_min_ex, y_max_ex = 0.0, 0.0 # Force 1D spatial for simplicity
    z_min_ex, z_max_ex = 0.0, 0.0 # Force 1D spatial for simplicity

    # Update config with these bounds
    config = PINNConfig(
        n_epochs=201, # Run for 201 epochs to see logging at 0 and 200
        learning_rate=1e-3,
        t_min=t_min_ex, t_max=t_max_ex,
        x_min=x_min_ex, x_max=x_max_ex,
        y_min=y_min_ex, y_max=y_max_ex,
        z_min=z_min_ex, z_max=z_max_ex
    )
    model, psi_model, trainer = create_pinn_system(key, config)

    # --- Generate Dummy Data (1D Wave Example: u(x,t) = sin(pi*x) * exp(-t)) ---
    n_data_pts = 100
    n_ic_pts = 50
    n_bc_pts = 50

    key, subkey_data_t, subkey_data_x, subkey_ic_x, subkey_bc_t, subkey_u_noise = random.split(trainer.key, 6) # Added key for noise
    trainer.key = key # Update trainer key state

    # Generate coordinates first
    t_data_coords = random.uniform(subkey_data_t, (n_data_pts,), minval=t_min_ex, maxval=t_max_ex)
    x_data_coords = random.uniform(subkey_data_x, (n_data_pts,), minval=x_min_ex, maxval=x_max_ex)
    x_ic_coords = random.uniform(subkey_ic_x, (n_ic_pts,), minval=x_min_ex, maxval=x_max_ex)
    t_bc_coords = random.uniform(subkey_bc_t, (n_bc_pts * 2,), minval=t_min_ex, maxval=t_max_ex)

    dummy_data = {
        # Random interior points
        't_data': t_data_coords,
        'x_data': x_data_coords,
        'y_data': jnp.zeros(n_data_pts),
        'z_data': jnp.zeros(n_data_pts),
        # Target u based on the example solution + small noise
        'u_data': jnp.sin(jnp.pi * x_data_coords) * jnp.exp(-t_data_coords) + \
                  0.01 * random.normal(subkey_u_noise, (n_data_pts,)),

        # Initial condition points (t=0)
        't_ic': jnp.zeros(n_ic_pts),
        'x_ic': x_ic_coords,
        'y_ic': jnp.zeros(n_ic_pts),
        'z_ic': jnp.zeros(n_ic_pts),
        'u_ic': jnp.sin(jnp.pi * x_ic_coords), # u(x, 0) = sin(pi*x)

        # Boundary condition points (x=-1 and x=1 for all t) - Dirichlet BC u=0
        't_bc': t_bc_coords,
        'x_bc': jnp.concatenate([jnp.full((n_bc_pts,), x_min_ex), jnp.full((n_bc_pts,), x_max_ex)]), # x=-1 and x=1
        'y_bc': jnp.zeros(n_bc_pts * 2),
        'z_bc': jnp.zeros(n_bc_pts * 2),
        'u_bc': jnp.zeros(n_bc_pts * 2) # u=0 at boundaries
    }

    # --- Train with Default Modulated Physics ---
    logger.info("--- Training with Default Modulated Wave Equation ---")
    try:
        model, psi_model = trainer.train(dummy_data) # Uses epochs from config (201)

        # --- Update with a Discovered Symbolic Equation ---
        # Example: A damped wave equation with forcing, modulated by psi
        new_eq_str = "u_tt - modulated_c**2 * u_xx + 0.1 * u_t - 5.0 * sin(2 * pi * x) * exp(-t) * (1 + tanh(psi))"
        compiled_fn, required_vars = compile_symbolic_to_jax(new_eq_str)

        if compiled_fn:
            trainer.set_symbolic_equation(compiled_fn, required_vars)

            # --- Continue Training with New Physics ---
            logger.info("--- Continuing Training with Symbolic Equation ---")
            model, psi_model = trainer.train(dummy_data, epochs=101) # Train a bit more (101 epochs)

    except Exception as e:
        logger.error(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    logger.info("--- Example Script Finished ---")
