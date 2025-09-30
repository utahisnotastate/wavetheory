"""
Advanced Visualization Modes
3D, heatmaps, field visualization, and interactive plots
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import streamlit as st
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for advanced visualizations."""
    resolution: int = 50
    color_scale: str = "Viridis"
    opacity: float = 0.8
    show_trails: bool = True
    trail_length: int = 50
    field_visualization: bool = True
    heatmap_mode: bool = False
    animation_frames: int = 100

class AdvancedVisualizer:
    """Advanced visualization system for Wave Theory simulations."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.field_cache = {}
    
    def create_3d_field_visualization(self, particles: List[Dict], 
                                    physics_params: Dict[str, float],
                                    domain: Dict[str, List[float]] = None) -> go.Figure:
        """Create 3D field visualization showing force fields."""
        if domain is None:
            domain = {"x": [-20, 20], "y": [-20, 20], "z": [-20, 20]}
        
        # Create grid
        x_range = np.linspace(domain["x"][0], domain["x"][1], self.config.resolution)
        y_range = np.linspace(domain["y"][0], domain["y"][1], self.config.resolution)
        z_range = np.linspace(domain["z"][0], domain["z"][1], self.config.resolution)
        
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Calculate field strength at each point
        field_strength = np.zeros_like(X)
        
        for particle in particles:
            px, py, pz = particle['position']
            mass = particle['mass']
            
            # Calculate distance from each grid point to particle
            dx = X - px
            dy = Y - py
            dz = Z - pz
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Avoid division by zero
            r = np.where(r < 0.1, 0.1, r)
            
            # Calculate Wave Theory force field
            G = physics_params.get('G', 1.0)
            wave_freq = physics_params.get('wave_frequency', 0.5)
            decay_length = physics_params.get('decay_length', 10.0)
            
            force_magnitude = G * mass / (r**2) * np.sin(wave_freq * r) * np.exp(-r / decay_length)
            field_strength += force_magnitude
        
        # Create 3D isosurface
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=field_strength.flatten(),
            isomin=field_strength.min(),
            isomax=field_strength.max(),
            surface_count=10,
            colorscale=self.config.color_scale,
            opacity=self.config.opacity,
            name="Force Field"
        ))
        
        # Add particles
        for i, particle in enumerate(particles):
            px, py, pz = particle['position']
            mass = particle['mass']
            color = particle.get('color', '#ff0000')
            
            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz],
                mode='markers',
                marker=dict(
                    size=mass * 2,
                    color=color,
                    opacity=1.0
                ),
                name=f"Particle {i+1}"
            ))
        
        fig.update_layout(
            title="3D Force Field Visualization",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_heatmap_visualization(self, particles: List[Dict], 
                                   physics_params: Dict[str, float],
                                   domain: Dict[str, List[float]] = None) -> go.Figure:
        """Create 2D heatmap visualization of field strength."""
        if domain is None:
            domain = {"x": [-20, 20], "y": [-20, 20]}
        
        # Create 2D grid
        x_range = np.linspace(domain["x"][0], domain["x"][1], self.config.resolution)
        y_range = np.linspace(domain["y"][0], domain["y"][1], self.config.resolution)
        
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        
        # Calculate field strength
        field_strength = np.zeros_like(X)
        
        for particle in particles:
            px, py = particle['position'][:2]  # Use only x, y coordinates
            mass = particle['mass']
            
            dx = X - px
            dy = Y - py
            r = np.sqrt(dx**2 + dy**2)
            r = np.where(r < 0.1, 0.1, r)
            
            G = physics_params.get('G', 1.0)
            wave_freq = physics_params.get('wave_frequency', 0.5)
            decay_length = physics_params.get('decay_length', 10.0)
            
            force_magnitude = G * mass / (r**2) * np.sin(wave_freq * r) * np.exp(-r / decay_length)
            field_strength += force_magnitude
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=field_strength,
            x=x_range,
            y=y_range,
            colorscale=self.config.color_scale,
            showscale=True,
            name="Field Strength"
        ))
        
        # Add particle positions
        for i, particle in enumerate(particles):
            px, py = particle['position'][:2]
            mass = particle['mass']
            color = particle.get('color', '#ff0000')
            
            fig.add_trace(go.Scatter(
                x=[px], y=[py],
                mode='markers',
                marker=dict(
                    size=mass * 3,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=f"Particle {i+1}"
            ))
        
        fig.update_layout(
            title="Field Strength Heatmap",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=500
        )
        
        return fig
    
    def create_particle_trajectory_plot(self, particles: List[Dict], 
                                      time_series: List[Dict]) -> go.Figure:
        """Create 3D trajectory plot with particle trails."""
        fig = go.Figure()
        
        for i, particle in enumerate(particles):
            color = particle.get('color', '#ff0000')
            
            # Extract trajectory for this particle
            trajectory_x = []
            trajectory_y = []
            trajectory_z = []
            
            for time_step in time_series:
                if i < len(time_step.get('particles', [])):
                    pos = time_step['particles'][i]['position']
                    trajectory_x.append(pos[0])
                    trajectory_y.append(pos[1])
                    trajectory_z.append(pos[2])
            
            if trajectory_x:
                # Add trajectory line
                fig.add_trace(go.Scatter3d(
                    x=trajectory_x,
                    y=trajectory_y,
                    z=trajectory_z,
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f"Particle {i+1} Trail",
                    opacity=0.7
                ))
                
                # Add start and end markers
                fig.add_trace(go.Scatter3d(
                    x=[trajectory_x[0]], y=[trajectory_y[0]], z=[trajectory_z[0]],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='circle'),
                    name=f"Particle {i+1} Start"
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=[trajectory_x[-1]], y=[trajectory_y[-1]], z=[trajectory_z[-1]],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='diamond'),
                    name=f"Particle {i+1} End"
                ))
        
        fig.update_layout(
            title="Particle Trajectories",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_energy_evolution_plot(self, time_series: List[Dict]) -> go.Figure:
        """Create energy evolution plot with multiple energy types."""
        if not time_series:
            return go.Figure()
        
        times = [step.get('time', 0) for step in time_series]
        kinetic_energy = [step.get('energy', {}).get('kinetic', 0) for step in time_series]
        potential_energy = [step.get('energy', {}).get('potential', 0) for step in time_series]
        total_energy = [step.get('energy', {}).get('total', 0) for step in time_series]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times, y=kinetic_energy,
            mode='lines+markers',
            name='Kinetic Energy',
            line=dict(color='#ff6b6b', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=potential_energy,
            mode='lines+markers',
            name='Potential Energy',
            line=dict(color='#4ecdc4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=total_energy,
            mode='lines+markers',
            name='Total Energy',
            line=dict(color='#45b7d1', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Energy Evolution",
            xaxis_title="Time",
            yaxis_title="Energy",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_force_vector_field(self, particles: List[Dict], 
                                physics_params: Dict[str, float],
                                domain: Dict[str, List[float]] = None) -> go.Figure:
        """Create vector field visualization of forces."""
        if domain is None:
            domain = {"x": [-20, 20], "y": [-20, 20]}
        
        # Create grid for vectors
        x_range = np.linspace(domain["x"][0], domain["x"][1], 20)
        y_range = np.linspace(domain["y"][0], domain["y"][1], 20)
        
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        
        # Calculate force vectors
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for particle in particles:
            px, py = particle['position'][:2]
            mass = particle['mass']
            
            dx = X - px
            dy = Y - py
            r = np.sqrt(dx**2 + dy**2)
            r = np.where(r < 0.1, 0.1, r)
            
            G = physics_params.get('G', 1.0)
            wave_freq = physics_params.get('wave_frequency', 0.5)
            decay_length = physics_params.get('decay_length', 10.0)
            
            force_magnitude = G * mass / (r**2) * np.sin(wave_freq * r) * np.exp(-r / decay_length)
            
            U += force_magnitude * (dx / r)
            V += force_magnitude * (dy / r)
        
        # Create vector field
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=Y.flatten(),
            mode='markers',
            marker=dict(size=1, color='lightgray'),
            name='Grid Points'
        ))
        
        # Add vector field
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=Y.flatten(),
            mode='markers+text',
            marker=dict(
                size=8,
                color=np.sqrt(U**2 + V**2).flatten(),
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"({u:.2f}, {v:.2f})" for u, v in zip(U.flatten(), V.flatten())],
            textposition="top center",
            name='Force Vectors'
        ))
        
        # Add particles
        for i, particle in enumerate(particles):
            px, py = particle['position'][:2]
            mass = particle['mass']
            color = particle.get('color', '#ff0000')
            
            fig.add_trace(go.Scatter(
                x=[px], y=[py],
                mode='markers',
                marker=dict(size=mass * 5, color=color),
                name=f"Particle {i+1}"
            ))
        
        fig.update_layout(
            title="Force Vector Field",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=500
        )
        
        return fig
    
    def create_animated_simulation(self, particles: List[Dict], 
                                 time_series: List[Dict]) -> go.Figure:
        """Create animated simulation visualization."""
        if not time_series:
            return go.Figure()
        
        # Prepare animation data
        frames = []
        
        for frame_idx, time_step in enumerate(time_series):
            frame_particles = time_step.get('particles', [])
            
            frame_data = []
            for i, particle in enumerate(frame_particles):
                pos = particle['position']
                mass = particle['mass']
                color = particle.get('color', '#ff0000')
                
                frame_data.append(go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode='markers',
                    marker=dict(size=mass * 3, color=color),
                    name=f"Particle {i+1}"
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=f"frame_{frame_idx}"
            ))
        
        # Create initial figure
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Animated Simulation",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100, 'redraw': True}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}}]
                    }
                ]
            }],
            height=600
        )
        
        return fig
    
    def create_phase_space_plot(self, particles: List[Dict], 
                              time_series: List[Dict]) -> go.Figure:
        """Create phase space plot (position vs velocity)."""
        if not time_series:
            return go.Figure()
        
        fig = go.Figure()
        
        for i, particle in enumerate(particles):
            color = particle.get('color', '#ff0000')
            
            positions = []
            velocities = []
            
            for time_step in time_series:
                if i < len(time_step.get('particles', [])):
                    particle_data = time_step['particles'][i]
                    pos = particle_data['position']
                    vel = particle_data['velocity']
                    
                    # Use magnitude for phase space
                    position_mag = np.sqrt(sum(p**2 for p in pos))
                    velocity_mag = np.sqrt(sum(v**2 for v in vel))
                    
                    positions.append(position_mag)
                    velocities.append(velocity_mag)
            
            if positions:
                fig.add_trace(go.Scatter(
                    x=positions, y=velocities,
                    mode='lines+markers',
                    name=f"Particle {i+1}",
                    line=dict(color=color, width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title="Phase Space Plot (Position vs Velocity)",
            xaxis_title="Position Magnitude",
            yaxis_title="Velocity Magnitude",
            height=400
        )
        
        return fig
    
    def create_spectral_analysis_plot(self, time_series: List[Dict]) -> go.Figure:
        """Create spectral analysis plot (FFT of particle motion)."""
        if not time_series:
            return go.Figure()
        
        fig = go.Figure()
        
        for i in range(len(time_series[0].get('particles', []))):
            positions = []
            times = []
            
            for time_step in time_series:
                if i < len(time_step.get('particles', [])):
                    particle_data = time_step['particles'][i]
                    pos = particle_data['position']
                    time = time_step.get('time', 0)
                    
                    # Use x-coordinate for spectral analysis
                    positions.append(pos[0])
                    times.append(time)
            
            if len(positions) > 1:
                # Perform FFT
                fft = np.fft.fft(positions)
                freqs = np.fft.fftfreq(len(positions), times[1] - times[0])
                
                # Plot power spectrum
                power_spectrum = np.abs(fft)**2
                
                fig.add_trace(go.Scatter(
                    x=freqs[:len(freqs)//2],
                    y=power_spectrum[:len(power_spectrum)//2],
                    mode='lines',
                    name=f"Particle {i+1}",
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Spectral Analysis (Power Spectrum)",
            xaxis_title="Frequency",
            yaxis_title="Power",
            height=400,
            xaxis=dict(type="log"),
            yaxis=dict(type="log")
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, particles: List[Dict], 
                                     time_series: List[Dict],
                                     physics_params: Dict[str, float]) -> Dict[str, go.Figure]:
        """Create comprehensive visualization dashboard."""
        dashboard = {}
        
        # 3D Field Visualization
        dashboard['field_3d'] = self.create_3d_field_visualization(particles, physics_params)
        
        # Heatmap
        dashboard['heatmap'] = self.create_heatmap_visualization(particles, physics_params)
        
        # Trajectory Plot
        dashboard['trajectories'] = self.create_particle_trajectory_plot(particles, time_series)
        
        # Energy Evolution
        dashboard['energy'] = self.create_energy_evolution_plot(time_series)
        
        # Force Vector Field
        dashboard['vector_field'] = self.create_force_vector_field(particles, physics_params)
        
        # Phase Space
        dashboard['phase_space'] = self.create_phase_space_plot(particles, time_series)
        
        # Spectral Analysis
        dashboard['spectral'] = self.create_spectral_analysis_plot(time_series)
        
        return dashboard

# Global visualizer instance
advanced_visualizer = AdvancedVisualizer()
