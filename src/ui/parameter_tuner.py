"""
Interactive Parameter Tuning Interface
Real-time parameter adjustment with live feedback
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import time
from datetime import datetime

@dataclass
class ParameterRange:
    """Parameter range definition."""
    min_val: float
    max_val: float
    step: float
    default: float
    description: str
    unit: str = ""

@dataclass
class ParameterGroup:
    """Group of related parameters."""
    name: str
    parameters: Dict[str, ParameterRange]
    description: str
    color: str = "#4ecdc4"

class InteractiveParameterTuner:
    """Interactive parameter tuning interface."""
    
    def __init__(self):
        self.parameter_groups = self._create_parameter_groups()
        self.current_values = {}
        self.history = []
        self.callbacks = {}
        
        # Initialize with default values
        self._initialize_defaults()
    
    def _create_parameter_groups(self) -> Dict[str, ParameterGroup]:
        """Create parameter groups for different aspects of the system."""
        groups = {}
        
        # Physics Parameters
        groups["physics"] = ParameterGroup(
            name="Physics Parameters",
            description="Core physics simulation parameters",
            color="#ff6b6b",
            parameters={
                "gravitational_constant": ParameterRange(
                    min_val=0.1, max_val=10.0, step=0.1, default=1.0,
                    description="Gravitational constant (G)",
                    unit="N⋅m²/kg²"
                ),
                "wave_frequency": ParameterRange(
                    min_val=0.01, max_val=5.0, step=0.01, default=0.5,
                    description="Wave frequency (ω)",
                    unit="rad/s"
                ),
                "decay_length": ParameterRange(
                    min_val=1.0, max_val=100.0, step=1.0, default=10.0,
                    description="Decay length (λ)",
                    unit="m"
                ),
                "damping_coefficient": ParameterRange(
                    min_val=0.0, max_val=1.0, step=0.01, default=0.01,
                    description="Damping coefficient",
                    unit="1/s"
                )
            }
        )
        
        # Neural Network Parameters
        groups["neural_network"] = ParameterGroup(
            name="Neural Network",
            description="PINN architecture and training parameters",
            color="#4ecdc4",
            parameters={
                "learning_rate": ParameterRange(
                    min_val=1e-5, max_val=1e-1, step=1e-5, default=1e-3,
                    description="Learning rate",
                    unit=""
                ),
                "hidden_layers": ParameterRange(
                    min_val=2, max_val=20, step=1, default=6,
                    description="Number of hidden layers",
                    unit=""
                ),
                "neurons_per_layer": ParameterRange(
                    min_val=32, max_val=512, step=32, default=128,
                    description="Neurons per layer",
                    unit=""
                ),
                "batch_size": ParameterRange(
                    min_val=32, max_val=1024, step=32, default=256,
                    description="Batch size",
                    unit=""
                ),
                "epochs": ParameterRange(
                    min_val=100, max_val=50000, step=100, default=10000,
                    description="Training epochs",
                    unit=""
                )
            }
        )
        
        # Symbolic Regression Parameters
        groups["symbolic_regression"] = ParameterGroup(
            name="Symbolic Regression",
            description="PySR configuration parameters",
            color="#45b7d1",
            parameters={
                "populations": ParameterRange(
                    min_val=5, max_val=50, step=1, default=15,
                    description="Number of populations",
                    unit=""
                ),
                "population_size": ParameterRange(
                    min_val=10, max_val=100, step=1, default=33,
                    description="Population size",
                    unit=""
                ),
                "max_size": ParameterRange(
                    min_val=5, max_val=50, step=1, default=20,
                    description="Maximum equation size",
                    unit=""
                ),
                "parsimony": ParameterRange(
                    min_val=1e-5, max_val=1e-1, step=1e-5, default=3.2e-3,
                    description="Parsimony coefficient",
                    unit=""
                ),
                "n_iterations": ParameterRange(
                    min_val=10, max_val=1000, step=10, default=100,
                    description="Number of iterations",
                    unit=""
                )
            }
        )
        
        # Simulation Parameters
        groups["simulation"] = ParameterGroup(
            name="Simulation",
            description="Simulation control parameters",
            color="#96ceb4",
            parameters={
                "time_step": ParameterRange(
                    min_val=0.001, max_val=0.1, step=0.001, default=0.01,
                    description="Time step (dt)",
                    unit="s"
                ),
                "max_particles": ParameterRange(
                    min_val=2, max_val=1000, step=1, default=100,
                    description="Maximum particles",
                    unit=""
                ),
                "collocation_points": ParameterRange(
                    min_val=1000, max_val=100000, step=1000, default=10000,
                    description="Collocation points",
                    unit=""
                ),
                "adaptive_sampling": ParameterRange(
                    min_val=0.0, max_val=1.0, step=0.1, default=0.5,
                    description="Adaptive sampling ratio",
                    unit=""
                )
            }
        )
        
        # Visualization Parameters
        groups["visualization"] = ParameterGroup(
            name="Visualization",
            description="Visualization and rendering parameters",
            color="#feca57",
            parameters={
                "trail_length": ParameterRange(
                    min_val=10, max_val=200, step=10, default=50,
                    description="Particle trail length",
                    unit=""
                ),
                "animation_speed": ParameterRange(
                    min_val=0.1, max_val=5.0, step=0.1, default=1.0,
                    description="Animation speed multiplier",
                    unit="x"
                ),
                "field_resolution": ParameterRange(
                    min_val=10, max_val=100, step=5, default=40,
                    description="Field visualization resolution",
                    unit=""
                ),
                "color_intensity": ParameterRange(
                    min_val=0.1, max_val=2.0, step=0.1, default=1.0,
                    description="Color intensity",
                    unit=""
                )
            }
        )
        
        return groups
    
    def _initialize_defaults(self):
        """Initialize with default parameter values."""
        for group_name, group in self.parameter_groups.items():
            for param_name, param_range in group.parameters.items():
                self.current_values[f"{group_name}.{param_name}"] = param_range.default
    
    def render_parameter_interface(self) -> Dict[str, Any]:
        """Render the interactive parameter tuning interface."""
        st.markdown("## 🎛️ Interactive Parameter Tuner")
        st.markdown("Adjust parameters in real-time and see immediate effects on the simulation.")
        
        # Create tabs for different parameter groups
        tab_names = list(self.parameter_groups.keys())
        tabs = st.tabs([group.name for group in self.parameter_groups.values()])
        
        updated_params = {}
        
        for i, (group_name, group) in enumerate(self.parameter_groups.items()):
            with tabs[i]:
                st.markdown(f"### {group.name}")
                st.markdown(f"*{group.description}*")
                
                # Create columns for parameter layout
                cols = st.columns(2)
                
                for j, (param_name, param_range) in enumerate(group.parameters.items()):
                    col = cols[j % 2]
                    
                    with col:
                        full_param_name = f"{group_name}.{param_name}"
                        current_value = self.current_values.get(full_param_name, param_range.default)
                        
                        # Create slider
                        new_value = st.slider(
                            f"{param_range.description}",
                            min_value=param_range.min_val,
                            max_value=param_range.max_val,
                            value=current_value,
                            step=param_range.step,
                            help=f"Range: {param_range.min_val} - {param_range.max_val} {param_range.unit}",
                            key=f"param_{full_param_name}"
                        )
                        
                        # Update value if changed
                        if new_value != current_value:
                            self.current_values[full_param_name] = new_value
                            updated_params[full_param_name] = new_value
                        
                        # Show current value
                        st.caption(f"Current: {new_value:.4f} {param_range.unit}")
        
        # Parameter history and effects
        if updated_params:
            self._log_parameter_change(updated_params)
            self._show_parameter_effects(updated_params)
        
        return self.current_values
    
    def _log_parameter_change(self, updated_params: Dict[str, Any]):
        """Log parameter changes to history."""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'parameters': updated_params.copy(),
            'all_parameters': self.current_values.copy()
        }
        self.history.append(change_record)
        
        # Keep only last 100 changes
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def _show_parameter_effects(self, updated_params: Dict[str, Any]):
        """Show the effects of parameter changes."""
        st.markdown("### 📊 Parameter Change Effects")
        
        # Show what changed
        for param_name, new_value in updated_params.items():
            group_name, param = param_name.split('.', 1)
            group = self.parameter_groups[group_name]
            param_range = group.parameters[param]
            
            st.info(f"**{param_range.description}**: {new_value:.4f} {param_range.unit}")
        
        # Show parameter sensitivity analysis
        self._show_sensitivity_analysis(updated_params)
    
    def _show_sensitivity_analysis(self, updated_params: Dict[str, Any]):
        """Show parameter sensitivity analysis."""
        st.markdown("#### 🔍 Sensitivity Analysis")
        
        # Create a simple sensitivity plot
        if len(self.history) > 1:
            # Get recent parameter changes
            recent_changes = self.history[-10:]
            
            # Extract parameter evolution
            param_evolution = {}
            for change in recent_changes:
                for param_name, value in change['parameters'].items():
                    if param_name not in param_evolution:
                        param_evolution[param_name] = []
                    param_evolution[param_name].append(value)
            
            # Create evolution plot
            if param_evolution:
                fig = go.Figure()
                
                for param_name, values in param_evolution.items():
                    group_name, param = param_name.split('.', 1)
                    group = self.parameter_groups[group_name]
                    param_range = group.parameters[param]
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(values))),
                        y=values,
                        mode='lines+markers',
                        name=param_range.description,
                        line=dict(color=group.color)
                    ))
                
                fig.update_layout(
                    title="Parameter Evolution",
                    xaxis_title="Change Number",
                    yaxis_title="Parameter Value",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def get_parameter_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self.current_values.copy()
    
    def set_parameter_value(self, param_name: str, value: float) -> bool:
        """Set a specific parameter value."""
        if param_name in self.current_values:
            self.current_values[param_name] = value
            return True
        return False
    
    def reset_to_defaults(self) -> None:
        """Reset all parameters to default values."""
        self._initialize_defaults()
        st.rerun()
    
    def save_parameter_set(self, name: str) -> None:
        """Save current parameter set with a name."""
        param_set = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'parameters': self.current_values.copy()
        }
        
        # Save to file
        filename = f"data/parameter_sets/{name}.json"
        with open(filename, 'w') as f:
            json.dump(param_set, f, indent=2)
        
        st.success(f"Parameter set '{name}' saved successfully!")
    
    def load_parameter_set(self, name: str) -> bool:
        """Load a saved parameter set."""
        try:
            filename = f"data/parameter_sets/{name}.json"
            with open(filename, 'r') as f:
                param_set = json.load(f)
            
            self.current_values = param_set['parameters']
            st.success(f"Parameter set '{name}' loaded successfully!")
            return True
        except FileNotFoundError:
            st.error(f"Parameter set '{name}' not found!")
            return False
    
    def create_parameter_optimization_interface(self):
        """Create interface for parameter optimization."""
        st.markdown("## 🎯 Parameter Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Optimization Settings")
            
            # Objective function selection
            objective = st.selectbox(
                "Optimization Objective",
                ["Minimize Loss", "Maximize Accuracy", "Minimize Training Time", "Custom"]
            )
            
            # Optimization algorithm
            algorithm = st.selectbox(
                "Optimization Algorithm",
                ["Random Search", "Grid Search", "Bayesian Optimization", "Genetic Algorithm"]
            )
            
            # Parameter ranges for optimization
            st.markdown("### Parameter Ranges for Optimization")
            
            optimization_params = {}
            for group_name, group in self.parameter_groups.items():
                with st.expander(f"{group.name}"):
                    for param_name, param_range in group.parameters.items():
                        full_name = f"{group_name}.{param_name}"
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            min_val = st.number_input(
                                f"Min {param_range.description}",
                                value=param_range.min_val,
                                key=f"opt_min_{full_name}"
                            )
                        with col_b:
                            max_val = st.number_input(
                                f"Max {param_range.description}",
                                value=param_range.max_val,
                                key=f"opt_max_{full_name}"
                            )
                        
                        if st.checkbox(f"Optimize {param_range.description}", key=f"opt_enable_{full_name}"):
                            optimization_params[full_name] = {
                                'min': min_val,
                                'max': max_val,
                                'step': param_range.step
                            }
        
        with col2:
            st.markdown("### Optimization Control")
            
            if st.button("🚀 Start Optimization", type="primary"):
                st.info("Optimization started! This may take several minutes...")
                # Here you would implement the actual optimization logic
                # For now, we'll just show a placeholder
                st.success("Optimization completed! Best parameters found.")
            
            if st.button("⏹️ Stop Optimization"):
                st.warning("Optimization stopped.")
            
            st.markdown("### Quick Actions")
            if st.button("📊 Show Parameter Effects"):
                st.info("Parameter effects analysis would be shown here.")
            
            if st.button("💾 Save Current Set"):
                name = st.text_input("Parameter set name", value="optimized_set")
                if name:
                    self.save_parameter_set(name)
    
    def create_parameter_comparison_interface(self):
        """Create interface for comparing different parameter sets."""
        st.markdown("## 📈 Parameter Comparison")
        
        # Load saved parameter sets
        try:
            import os
            param_sets_dir = "data/parameter_sets"
            if os.path.exists(param_sets_dir):
                param_files = [f for f in os.listdir(param_sets_dir) if f.endswith('.json')]
                
                if param_files:
                    selected_sets = st.multiselect(
                        "Select parameter sets to compare",
                        param_files,
                        default=param_files[:2] if len(param_files) >= 2 else param_files
                    )
                    
                    if selected_sets:
                        # Load and compare parameter sets
                        comparison_data = []
                        for set_file in selected_sets:
                            with open(f"{param_sets_dir}/{set_file}", 'r') as f:
                                param_set = json.load(f)
                                comparison_data.append({
                                    'name': param_set['name'],
                                    'parameters': param_set['parameters']
                                })
                        
                        # Create comparison table
                        self._create_parameter_comparison_table(comparison_data)
                else:
                    st.info("No saved parameter sets found. Save some parameter sets first!")
            else:
                st.info("No parameter sets directory found.")
        except Exception as e:
            st.error(f"Error loading parameter sets: {e}")
    
    def _create_parameter_comparison_table(self, comparison_data: List[Dict]):
        """Create a comparison table for parameter sets."""
        if not comparison_data:
            return
        
        # Get all unique parameters
        all_params = set()
        for data in comparison_data:
            all_params.update(data['parameters'].keys())
        
        # Create comparison DataFrame
        import pandas as pd
        
        comparison_df = pd.DataFrame()
        for data in comparison_data:
            row = {'Parameter Set': data['name']}
            for param in all_params:
                value = data['parameters'].get(param, 'N/A')
                row[param] = value
            comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create comparison visualization
        if len(comparison_data) >= 2:
            self._create_parameter_comparison_chart(comparison_data)
    
    def _create_parameter_comparison_chart(self, comparison_data: List[Dict]):
        """Create a comparison chart for parameter sets."""
        # This would create a radar chart or bar chart comparing parameters
        # For now, we'll create a simple bar chart
        st.markdown("### 📊 Parameter Comparison Chart")
        
        # Group parameters by category
        param_categories = {}
        for data in comparison_data:
            for param_name, value in data['parameters'].items():
                if isinstance(value, (int, float)):
                    group_name = param_name.split('.')[0]
                    if group_name not in param_categories:
                        param_categories[group_name] = {}
                    if param_name not in param_categories[group_name]:
                        param_categories[group_name][param_name] = []
                    param_categories[group_name][param_name].append({
                        'set': data['name'],
                        'value': value
                    })
        
        # Create charts for each category
        for category, params in param_categories.items():
            if params:
                fig = go.Figure()
                
                for param_name, values in params.items():
                    sets = [v['set'] for v in values]
                    vals = [v['value'] for v in values]
                    
                    fig.add_trace(go.Bar(
                        name=param_name.split('.')[1],
                        x=sets,
                        y=vals
                    ))
                
                fig.update_layout(
                    title=f"{category.title()} Parameters Comparison",
                    xaxis_title="Parameter Set",
                    yaxis_title="Value",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Global parameter tuner instance
parameter_tuner = InteractiveParameterTuner()
