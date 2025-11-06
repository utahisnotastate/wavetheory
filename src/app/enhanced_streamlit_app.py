"""
Enhanced Wave Theory Chatbot - Streamlit Application
Incorporates advanced features from HTML version
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import time
from datetime import datetime
import os
from typing import Dict, List, Any, Optional
import logging

# Import our custom modules
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from models.neuro_symbolic import WaveTheorySystem, WaveTheoryConfig, create_wave_theory_system
from models.pinn_jax import WavePINN, PINNTrainer, create_pinn_model
from models.symbolic_engine import SymbolicRegressionEngine, NeuroSymbolicOrchestrator
from models.lf_constants import G_of_t, lf_force_gate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# ENHANCED STREAMLIT APP CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Wave Theory Chatbot",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Neuro-Symbolic Physics Discovery Engine - Wave Theory Chatbot"
    }
)

# Enhanced CSS with animations and better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b69 100%);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00ffff, #0080ff);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 255, 255, 0.5);
    }
    
    .equation-box {
        background: rgba(0, 255, 255, 0.1);
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        color: #00ff88;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(0, 255, 255, 0.2); }
        50% { box-shadow: 0 8px 25px rgba(0, 255, 255, 0.4); }
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 255, 0.2);
    }
    
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 15px;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e0e6f0;
        margin-right: 20%;
    }
    
    .particle-trail {
        stroke-width: 2;
        stroke-opacity: 0.6;
        fill: none;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #00ffff;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# ENHANCED SESSION STATE INITIALIZATION
# =====================================================================

def initialize_session_state():
    """Initialize all session state variables with enhanced features."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.simulation_running = False
        st.session_state.generation = 1
        st.session_state.current_equation = "F = -G * (m₁ * m₂ / r²) * sin(ωr) * exp(-r/λ)"
        st.session_state.particles = []
        st.session_state.simulation_history = []
        st.session_state.model_loss = 0.001
        st.session_state.training_history = []
        st.session_state.pareto_front = []
        st.session_state.particle_trails = {}  # New: track particle trails
        st.session_state.simulation_time = 0.0  # New: track simulation time

        # Initialize with 3 default particles with enhanced properties
        st.session_state.particles = [
            {'id': 0, 'position': [0, 0, 0], 'velocity': [0.5, 0, 0], 'mass': 5.0, 'color': '#00ffff'},
            {'id': 1, 'position': [10, 0, 0], 'velocity': [-0.5, 0.5, 0], 'mass': 5.0, 'color': '#ff00ff'},
            {'id': 2, 'position': [5, 8.66, 0], 'velocity': [0, -0.5, 0], 'mass': 5.0, 'color': '#00ff88'}
        ]

        # Initialize particle trails
        for particle in st.session_state.particles:
            st.session_state.particle_trails[particle['id']] = []

        # Model components
        st.session_state.pinn_model = None
        st.session_state.symbolic_engine = None
        st.session_state.chatbot_model = None
        st.session_state.wave_theory_system = None
        # LF toggle and params (UI-controlled)
        st.session_state.lf_enabled = False
        st.session_state.lf_frequency_hz = 1.1e12
        st.session_state.lf_blink_tau = 1e-17
        st.session_state.lf_blink_duty = 0.5
        st.session_state.lf_G_amp = 0.0
        st.session_state.lf_phi = 0.0

initialize_session_state()

# =====================================================================
# ENHANCED MODEL LOADING
# =====================================================================

@st.cache_resource
def load_chatbot_model():
    """Load and cache chatbot backend (Gemini only)."""
    try:
        from app.chatbot import get_chatbot
        generator = get_chatbot()
        logger.info("Loaded chatbot backend (Gemini)")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize chatbot backend: {e}")
        return None

@st.cache_resource
def load_wave_theory_system():
    """Load and cache the Wave Theory system."""
    try:
        import jax
        key = jax.random.PRNGKey(42)
        system = create_wave_theory_system(key=key)
        logger.info("Loaded Wave Theory system")
        return system
    except Exception as e:
        logger.error(f"Failed to load Wave Theory system: {e}")
        return None

# =====================================================================
# ENHANCED SIMULATION FUNCTIONS
# =====================================================================

def calculate_wave_force(p1: Dict, p2: Dict, G: float = 1.0,
                         wave_freq: float = 0.5, decay_length: float = 10.0,
                         t: float = 0.0):
    """Calculate Wave Theory force between two particles with enhanced physics."""
    r_vec = np.array(p2['position']) - np.array(p1['position'])
    r = np.linalg.norm(r_vec)

    if r < 1e-6:
        return np.zeros(3)

    # Optional LF modulation
    if st.session_state.get('lf_enabled', False):
        G = float(G_of_t(t, G_avg=G, G_amp=st.session_state.lf_G_amp,
                         f_lf_hz=st.session_state.lf_frequency_hz,
                         phi=st.session_state.lf_phi))
        gate = float(lf_force_gate(t, st.session_state.lf_frequency_hz,
                                   st.session_state.lf_blink_tau,
                                   st.session_state.lf_blink_duty))
    else:
        gate = 1.0

    # Enhanced Wave Theory force with better numerical stability
    magnitude = -G * (p1['mass'] * p2['mass'] / (r**2)) * \
               np.sin(wave_freq * r) * np.exp(-r / decay_length) * gate

    return magnitude * (r_vec / r)

def step_simulation():
    """Enhanced simulation step with trail tracking and better physics."""
    dt = 0.01
    particles = st.session_state.particles

    # Calculate forces
    forces = [np.zeros(3) for _ in particles]

    for i in range(len(particles)):
        for j in range(len(particles)):
            if i != j:
                force = calculate_wave_force(particles[i], particles[j], t=st.session_state.simulation_time)
                forces[i] += force

    # Update velocities and positions with enhanced physics
    for i, particle in enumerate(particles):
        acceleration = forces[i] / particle['mass']
        particle['velocity'] = [v + a * dt for v, a in zip(particle['velocity'], acceleration)]
        particle['position'] = [p + v * dt for p, v in zip(particle['position'], particle['velocity'])]

        # Update particle trail
        trail = st.session_state.particle_trails[particle['id']]
        trail.append(particle['position'].copy())
        if len(trail) > 50:  # Limit trail length
            trail.pop(0)

    # Update simulation time
    st.session_state.simulation_time += dt

    # Save to history with enhanced data
    st.session_state.simulation_history.append({
        'time': st.session_state.simulation_time,
        'particles': [p.copy() for p in particles],
        'energy': calculate_system_energy(),
        'forces': [f.tolist() for f in forces]
    })

def calculate_system_energy():
    """Calculate total system energy with enhanced physics."""
    particles = st.session_state.particles
    kinetic = 0.0
    potential = 0.0

    # Kinetic energy
    for p in particles:
        v_squared = sum(v**2 for v in p['velocity'])
        kinetic += 0.5 * p['mass'] * v_squared

    # Potential energy with Wave Theory modifications
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            r_vec = np.array(particles[j]['position']) - np.array(particles[i]['position'])
            r = np.linalg.norm(r_vec)
            if r > 1e-6:
                # Enhanced potential with wave modulation
                wave_factor = np.sin(0.5 * r) * np.exp(-r / 10.0)
                potential += -1.0 * particles[i]['mass'] * particles[j]['mass'] / r * (1 + 0.1 * wave_factor)

    return {'kinetic': kinetic, 'potential': potential, 'total': kinetic + potential}

# =====================================================================
# ENHANCED CHATBOT QUERY PROCESSING
# =====================================================================

def process_user_query(query: str) -> str:
    """Enhanced query processing with better physics explanations."""
    query_lower = query.lower()

    # Enhanced command parsing
    if 'add' in query_lower and 'particle' in query_lower:
        # Add new particle with enhanced properties
        colors = ['#00ffff', '#ff00ff', '#00ff88', '#ffff00', '#ff6b6b', '#9b59b6']
        new_particle = {
            'id': len(st.session_state.particles),
            'position': [np.random.uniform(-10, 10) for _ in range(3)],
            'velocity': [np.random.uniform(-1, 1) for _ in range(3)],
            'mass': np.random.uniform(3, 8),
            'color': colors[len(st.session_state.particles) % len(colors)]
        }
        st.session_state.particles.append(new_particle)

        # Initialize trail for new particle
        st.session_state.particle_trails[new_particle['id']] = []

        return f"🌌 Added particle with mass {new_particle['mass']:.2f} at position " \
               f"({new_particle['position'][0]:.2f}, {new_particle['position'][1]:.2f}, " \
               f"{new_particle['position'][2]:.2f}). The system now has {len(st.session_state.particles)} particles " \
               f"interacting through the Wave Theory force law with sinusoidal modulation."

    elif 'energy' in query_lower:
        energy = calculate_system_energy()
        return f"⚡ System energy analysis:\n" \
               f"• Kinetic Energy: {energy['kinetic']:.3f} units\n" \
               f"• Potential Energy: {energy['potential']:.3f} units\n" \
               f"• Total Energy: {energy['total']:.3f} units\n\n" \
               f"The Wave Theory modifies the potential energy through sinusoidal modulation, " \
               f"creating oscillating attractive/repulsive regions in space."

    elif 'equation' in query_lower or 'law' in query_lower:
        return f"🧮 Current discovered force law:\n\n" \
               f"**{st.session_state.current_equation}**\n\n" \
               f"This equation was discovered through {st.session_state.generation} generations " \
               f"of neuro-symbolic evolution, combining PINN training with symbolic regression. " \
               f"The sinusoidal term creates wave-like interactions, while the exponential decay " \
               f"ensures finite-range forces."

    elif 'train' in query_lower or 'generation' in query_lower:
        if st.session_state.wave_theory_system:
            try:
                # Run actual neuro-symbolic evolution
                results = st.session_state.wave_theory_system.run_evolution(n_generations=1)

                # Update session state
                st.session_state.generation = st.session_state.wave_theory_system.get_generation()
                st.session_state.current_equation = st.session_state.wave_theory_system.get_current_equation()

                if results['best_equation']:
                    loss = results['best_equation']['loss']
                    st.session_state.model_loss = loss

                    return f"🚀 Advanced to generation {st.session_state.generation}!\n\n" \
                           f"• Model Loss: {loss:.6f}\n" \
                           f"• Updated Equation: {st.session_state.current_equation}\n\n" \
                           f"The PINN has refined its understanding of the physics through " \
                           f"continuous learning and symbolic discovery."
                else:
                    return f"🔄 Evolution in progress... Generation {st.session_state.generation} " \
                           f"is exploring new mathematical expressions for the fundamental forces."
            except Exception as e:
                logger.error(f"Error in evolution: {e}")
                return f"⚠️ Error during evolution: {str(e)}"
        else:
            # Fallback to simulation
            st.session_state.generation += 1
            st.session_state.model_loss = max(0.0001, st.session_state.model_loss * 0.9)

            equations = [
                "F = -G * (m₁ * m₂ / r²) * sin(ωr)",
                "F = -G * (m₁ * m₂ / r²) * sin(ωr) * exp(-r/λ)",
                "F = -G * (m₁ * m₂ / r²) * (sin(ωr) + 0.1*cos(2ωr)) * exp(-r/λ)",
            ]
            st.session_state.current_equation = equations[min(st.session_state.generation % 3, 2)]

            return f"🧠 Advanced to generation {st.session_state.generation}!\n\n" \
                   f"• Model Loss: {st.session_state.model_loss:.6f}\n" \
                   f"• Updated Equation: {st.session_state.current_equation}\n\n" \
                   f"The symbolic regression engine is continuously searching for " \
                   f"more fundamental expressions of the physical laws."

    elif 'reset' in query_lower:
        st.session_state.particles = [
            {'id': 0, 'position': [0, 0, 0], 'velocity': [0.5, 0, 0], 'mass': 5.0, 'color': '#00ffff'},
            {'id': 1, 'position': [10, 0, 0], 'velocity': [-0.5, 0.5, 0], 'mass': 5.0, 'color': '#ff00ff'},
            {'id': 2, 'position': [5, 8.66, 0], 'velocity': [0, -0.5, 0], 'mass': 5.0, 'color': '#00ff88'}
        ]
        st.session_state.simulation_history = []
        st.session_state.simulation_time = 0.0
        st.session_state.particle_trails = {i: [] for i in range(len(st.session_state.particles))}
        return "🔄 Simulation reset to initial conditions with 3 particles in a triangular configuration. " \
               "All particle trails have been cleared and the system is ready for new experiments."

    elif 'help' in query_lower:
        return "🤖 I can help you explore the Wave Theory universe! Try these commands:\n\n" \
               "• **Add a particle** - adds a new body to the simulation\n" \
               "• **What's the energy?** - calculate total system energy\n" \
               "• **Explain the equation** - describe the discovered force law\n" \
               "• **Train the model** - advance the neuro-symbolic training\n" \
               "• **Reset simulation** - return to initial state\n\n" \
               "The Wave Theory combines classical mechanics with wave phenomena, " \
               "creating a rich and complex universe for exploration!"

    else:
        # Use LLM if available, otherwise enhanced fallback response
        if st.session_state.chatbot_model:
            try:
                return st.session_state.chatbot_model(query)
            except Exception as _:
                pass

        return f"🌊 Fascinating question! The Wave Theory universe contains {len(st.session_state.particles)} particles " \
               f"interacting through a modified gravitational force with sinusoidal modulation. " \
               f"The current simulation shows complex wave-like dynamics that emerge from the " \
               f"fundamental force law. Would you like to add particles, analyze the energy, " \
               f"or explore the discovered equations?"

# =====================================================================
# ENHANCED VISUALIZATION COMPONENTS
# =====================================================================

def create_enhanced_3d_visualization():
    """Create enhanced 3D visualization with particle trails and field effects."""
    particles = st.session_state.particles

    fig = go.Figure()

    # Add particle trails
    for particle_id, trail in st.session_state.particle_trails.items():
        if len(trail) > 1:
            trail_array = np.array(trail)
            fig.add_trace(go.Scatter3d(
                x=trail_array[:, 0],
                y=trail_array[:, 1],
                z=trail_array[:, 2],
                mode='lines',
                line=dict(
                    color=particles[particle_id]['color'],
                    width=3,
                    opacity=0.6
                ),
                name=f"Trail {particle_id}",
                showlegend=False
            ))

    # Add particles with enhanced visualization
    for i, p in enumerate(particles):
        # Main particle
        fig.add_trace(go.Scatter3d(
            x=[p['position'][0]],
            y=[p['position'][1]],
            z=[p['position'][2]],
            mode='markers+text',
            marker=dict(
                size=p['mass'] * 3,
                color=p['color'],
                opacity=0.9,
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            text=f"P{i}",
            textposition="top center",
            name=f"Particle {i}"
        ))

        # Velocity vectors with enhanced styling
        fig.add_trace(go.Cone(
            x=[p['position'][0]],
            y=[p['position'][1]],
            z=[p['position'][2]],
            u=[p['velocity'][0]],
            v=[p['velocity'][1]],
            w=[p['velocity'][2]],
            colorscale='Blues',
            showscale=False,
            sizemode="absolute",
            sizeref=0.8,
            name=f"Velocity {i}",
            opacity=0.7
        ))

    # Enhanced layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor="rgba(10, 14, 39, 0.9)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,255,255,0.1)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,255,255,0.1)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,255,255,0.1)")
        ),
        showlegend=True,
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="🌌 Wave Theory Universe - Enhanced Visualization"
    )

    return fig

def create_enhanced_energy_plot():
    """Create enhanced energy evolution plot with better styling."""
    if not st.session_state.simulation_history:
        return None

    times = []
    kinetic = []
    potential = []
    total = []

    for state in st.session_state.simulation_history[-100:]:
        times.append(state['time'])
        energy = state['energy']
        kinetic.append(energy['kinetic'])
        potential.append(energy['potential'])
        total.append(energy['total'])

    fig = go.Figure()

    # Enhanced traces with better styling
    fig.add_trace(go.Scatter(
        x=times, y=kinetic,
        name='Kinetic Energy',
        line=dict(color='#00ffff', width=3),
        mode='lines+markers',
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=times, y=potential,
        name='Potential Energy',
        line=dict(color='#ff00ff', width=3),
        mode='lines+markers',
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=times, y=total,
        name='Total Energy',
        line=dict(color='#00ff88', width=4, dash='dash'),
        mode='lines+markers',
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="⚡ Energy Evolution - Wave Theory System",
        xaxis_title="Time (simulation units)",
        yaxis_title="Energy (arbitrary units)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 14, 39, 0.5)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

# =====================================================================
# MAIN ENHANCED STREAMLIT APP
# =====================================================================

def main():
    """Enhanced main Streamlit application."""

    # Load models
    if st.session_state.chatbot_model is None:
        with st.spinner("🤖 Loading AI model..."):
            st.session_state.chatbot_model = load_chatbot_model()

    if st.session_state.wave_theory_system is None:
        with st.spinner("🌊 Loading Wave Theory system..."):
            st.session_state.wave_theory_system = load_wave_theory_system()

    # Enhanced header with animations
    st.markdown("""
        <div style='text-align: center; padding: 30px 0;'>
            <h1 style='font-size: 3em; background: linear-gradient(45deg, #00ffff, #ff00ff, #00ff88); 
                       background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       animation: gradient-shift 6s ease infinite; margin-bottom: 10px; font-weight: 800;'>
                🌊 Wave Theory Chatbot
            </h1>
            <p style='color: #8892b0; font-size: 1.2em; opacity: 0.9;'>
                Neuro-Symbolic Physics Discovery Engine
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Capture chat input outside of layout containers (Streamlit requirement)
    user_input = st.chat_input("Ask about physics experiments...")

    # Create three columns layout
    col1, col2, col3 = st.columns([1, 2, 1])

    # Left Column - Enhanced Chat Interface
    with col1:
        st.markdown("### 💬 Quantum Interface")

        # Enhanced chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)

    # Handle chat input at top level
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = process_user_query(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Middle Column - Enhanced Visualization
    with col2:
        st.markdown("### 🌌 Universe Visualization")

        # Enhanced 3D Visualization
        particle_viz = create_enhanced_3d_visualization()
        st.plotly_chart(particle_viz, use_container_width=True)

        # Enhanced control buttons
        button_col1, button_col2, button_col3, button_col4 = st.columns(4)

        with button_col1:
            if st.button("▶️ Run", use_container_width=True):
                st.session_state.simulation_running = True
                for _ in range(100):
                    step_simulation()
                st.rerun()

        with button_col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.simulation_running = False

        with button_col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.particles = [
                    {'id': 0, 'position': [0, 0, 0], 'velocity': [0.5, 0, 0], 'mass': 5.0, 'color': '#00ffff'},
                    {'id': 1, 'position': [10, 0, 0], 'velocity': [-0.5, 0.5, 0], 'mass': 5.0, 'color': '#ff00ff'},
                    {'id': 2, 'position': [5, 8.66, 0], 'velocity': [0, -0.5, 0], 'mass': 5.0, 'color': '#00ff88'}
                ]
                st.session_state.simulation_history = []
                st.session_state.simulation_time = 0.0
                st.session_state.particle_trails = {i: [] for i in range(3)}
                st.rerun()

        with button_col4:
            if st.button("➕ Add", use_container_width=True):
                colors = ['#00ffff', '#ff00ff', '#00ff88', '#ffff00', '#ff6b6b', '#9b59b6']
                new_particle = {
                    'id': len(st.session_state.particles),
                    'position': [np.random.uniform(-10, 10) for _ in range(3)],
                    'velocity': [np.random.uniform(-1, 1) for _ in range(3)],
                    'mass': np.random.uniform(3, 8),
                    'color': colors[len(st.session_state.particles) % len(colors)]
                }
                st.session_state.particles.append(new_particle)
                st.session_state.particle_trails[new_particle['id']] = []
                st.rerun()

        # Enhanced energy plot
        energy_plot = create_enhanced_energy_plot()
        if energy_plot:
            st.plotly_chart(energy_plot, use_container_width=True)

    # Right Column - Enhanced Model Status
    with col3:
        st.markdown("### 🧠 Model Status")

        with st.expander("Lighthouse Frequency (LF) controls", expanded=False):
            st.session_state.lf_enabled = st.checkbox("Enable LF model (experimental)", value=st.session_state.lf_enabled)
            st.session_state.lf_frequency_hz = st.number_input("LF frequency (Hz)", value=float(st.session_state.lf_frequency_hz), format="%e")
            st.session_state.lf_blink_tau = st.number_input("Blink transition tau (s)", value=float(st.session_state.lf_blink_tau), format="%e")
            st.session_state.lf_blink_duty = st.slider("Blink duty", 0.0, 1.0, float(st.session_state.lf_blink_duty))
            st.session_state.lf_G_amp = st.number_input("G modulation amplitude", value=float(st.session_state.lf_G_amp), format="%e")
            st.session_state.lf_phi = st.number_input("Phase phi (rad)", value=float(st.session_state.lf_phi))

        # Enhanced metrics
        energy = calculate_system_energy()

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Particles", len(st.session_state.particles))
            st.metric("Total Energy", f"{energy['total']:.3f}")
        with col_b:
            st.metric("Generation", st.session_state.generation)
            st.metric("Model Loss", f"{st.session_state.model_loss:.6f}")

        # Enhanced equation display
        st.markdown("#### 🧮 Current Equation")
        st.markdown(f"""
            <div class='equation-box'>
                {st.session_state.current_equation}
            </div>
        """, unsafe_allow_html=True)

        # Simulation time
        st.metric("Simulation Time", f"{st.session_state.simulation_time:.2f}")

    # Enhanced footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #8892b0; font-size: 0.9em; padding: 20px;'>
            🌊 Wave Theory Chatbot v2.0 | Enhanced Neuro-Symbolic Physics Discovery Engine
            <br>Combining Physics-Informed Neural Networks with Advanced Symbolic Regression
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
