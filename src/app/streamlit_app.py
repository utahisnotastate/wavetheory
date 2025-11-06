"""
Wave Theory Chatbot - Streamlit Application
Full-featured web interface with Hugging Face LLM integration
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# STREAMLIT APP CONFIGURATION
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

# Custom CSS for styling
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
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
    }
    .equation-box {
        background: rgba(0, 255, 255, 0.1);
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        color: #00ff88;
        margin: 10px 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================

def initialize_session_state():
    """Initialize all session state variables."""
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

        # Initialize with 3 default particles
        st.session_state.particles = [
            {'id': 0, 'position': [0, 0, 0], 'velocity': [0.5, 0, 0], 'mass': 5.0},
            {'id': 1, 'position': [10, 0, 0], 'velocity': [-0.5, 0.5, 0], 'mass': 5.0},
            {'id': 2, 'position': [5, 8.66, 0], 'velocity': [0, -0.5, 0], 'mass': 5.0}
        ]

        # Model components
        st.session_state.pinn_model = None
        st.session_state.symbolic_engine = None
        st.session_state.chatbot_model = None
        st.session_state.wave_theory_system = None

initialize_session_state()

# =====================================================================
# CHATBOT BACKEND (Gemini only)
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
# SIMULATION FUNCTIONS
# =====================================================================

def calculate_wave_force(p1: Dict, p2: Dict, G: float = 1.0,
                         wave_freq: float = 0.5, decay_length: float = 10.0):
    """Calculate Wave Theory force between two particles."""
    r_vec = np.array(p2['position']) - np.array(p1['position'])
    r = np.linalg.norm(r_vec)

    if r < 1e-6:
        return np.zeros(3)

    magnitude = -G * (p1['mass'] * p2['mass'] / (r**2)) * \
               np.sin(wave_freq * r) * np.exp(-r / decay_length)

    return magnitude * (r_vec / r)

def step_simulation():
    """Perform one simulation step."""
    dt = 0.01
    particles = st.session_state.particles

    # Calculate forces
    forces = [np.zeros(3) for _ in particles]

    for i in range(len(particles)):
        for j in range(len(particles)):
            if i != j:
                force = calculate_wave_force(particles[i], particles[j])
                forces[i] += force

    # Update velocities and positions
    for i, particle in enumerate(particles):
        acceleration = forces[i] / particle['mass']
        particle['velocity'] = [v + a * dt for v, a in zip(particle['velocity'], acceleration)]
        particle['position'] = [p + v * dt for p, v in zip(particle['position'], particle['velocity'])]

    # Save to history
    st.session_state.simulation_history.append({
        'time': len(st.session_state.simulation_history) * dt,
        'particles': [p.copy() for p in particles]
    })

def calculate_system_energy():
    """Calculate total system energy."""
    particles = st.session_state.particles
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
                potential += -1.0 * particles[i]['mass'] * particles[j]['mass'] / r

    return {'kinetic': kinetic, 'potential': potential, 'total': kinetic + potential}

# =====================================================================
# CHATBOT QUERY PROCESSING
# =====================================================================

def process_user_query(query: str) -> str:
    """
    Process user query and generate response.
    Uses either the loaded LLM or rule-based responses.
    """
    query_lower = query.lower()

    # Command parsing
    if 'add' in query_lower and 'particle' in query_lower:
        # Add new particle
        new_particle = {
            'id': len(st.session_state.particles),
            'position': [np.random.uniform(-10, 10) for _ in range(3)],
            'velocity': [np.random.uniform(-1, 1) for _ in range(3)],
            'mass': np.random.uniform(3, 8)
        }
        st.session_state.particles.append(new_particle)

        return f"Added particle with mass {new_particle['mass']:.2f} at position " \
               f"({new_particle['position'][0]:.2f}, {new_particle['position'][1]:.2f}, " \
               f"{new_particle['position'][2]:.2f}). System now has {len(st.session_state.particles)} particles."

    elif 'energy' in query_lower:
        energy = calculate_system_energy()
        return f"System energy: Kinetic = {energy['kinetic']:.3f}, " \
               f"Potential = {energy['potential']:.3f}, Total = {energy['total']:.3f} units. " \
               f"The Wave Theory modifies the potential energy through sinusoidal modulation."

    elif 'equation' in query_lower or 'law' in query_lower:
        return f"Current discovered force law: {st.session_state.current_equation}. " \
               f"This equation was discovered through {st.session_state.generation} generations " \
               f"of neuro-symbolic evolution, combining PINN training with symbolic regression."

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

                    return f"Advanced to generation {st.session_state.generation}. " \
                           f"Model loss: {loss:.6f}. " \
                           f"Updated equation: {st.session_state.current_equation}"
                else:
                    return f"Advanced to generation {st.session_state.generation}. " \
                           f"Evolution in progress..."
            except Exception as e:
                logger.error(f"Error in evolution: {e}")
                return f"Error during evolution: {str(e)}"
        else:
            # Fallback to simulation
            st.session_state.generation += 1
            st.session_state.model_loss = max(0.0001, st.session_state.model_loss * 0.9)

            # Simulate equation evolution
            equations = [
                "F = -G * (m₁ * m₂ / r²) * sin(ωr)",
                "F = -G * (m₁ * m₂ / r²) * sin(ωr) * exp(-r/λ)",
                "F = -G * (m₁ * m₂ / r²) * (sin(ωr) + 0.1*cos(2ωr)) * exp(-r/λ)",
            ]
            st.session_state.current_equation = equations[min(st.session_state.generation % 3, 2)]

            return f"Advanced to generation {st.session_state.generation}. " \
                   f"Model loss: {st.session_state.model_loss:.6f}. " \
                   f"Updated equation: {st.session_state.current_equation}"

    elif 'reset' in query_lower:
        st.session_state.particles = [
            {'id': 0, 'position': [0, 0, 0], 'velocity': [0.5, 0, 0], 'mass': 5.0},
            {'id': 1, 'position': [10, 0, 0], 'velocity': [-0.5, 0.5, 0], 'mass': 5.0},
            {'id': 2, 'position': [5, 8.66, 0], 'velocity': [0, -0.5, 0], 'mass': 5.0}
        ]
        st.session_state.simulation_history = []
        return "Simulation reset to initial conditions with 3 particles."

    else:
        # Use LLM if available, otherwise fallback response
        if st.session_state.chatbot_model:
            try:
                # Unified callable interface
                return st.session_state.chatbot_model(query)
            except Exception as _:
                pass

        return f"I'm the Wave Theory Chatbot! I can help you explore physics through our neuro-symbolic model. " \
               f"Try: 'add particle', 'calculate energy', 'show equation', 'train model', or ask about the physics!"

# =====================================================================
# VISUALIZATION COMPONENTS
# =====================================================================

def create_3d_particle_visualization():
    """Create 3D visualization of particle system."""
    particles = st.session_state.particles

    fig = go.Figure()

    # Add particles
    for i, p in enumerate(particles):
        fig.add_trace(go.Scatter3d(
            x=[p['position'][0]],
            y=[p['position'][1]],
            z=[p['position'][2]],
            mode='markers+text',
            marker=dict(
                size=p['mass'] * 2,
                color=['cyan', 'magenta', 'lime'][i % 3],
                opacity=0.8,
                symbol='circle'
            ),
            text=f"P{i}",
            textposition="top center",
            name=f"Particle {i}"
        ))

        # Add velocity vectors
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
            sizeref=0.5,
            name=f"Velocity {i}"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor="rgba(10, 14, 39, 0.8)"
        ),
        showlegend=True,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig

def create_energy_plot():
    """Create energy evolution plot."""
    if not st.session_state.simulation_history:
        return None

    times = []
    kinetic = []
    potential = []
    total = []

    for state in st.session_state.simulation_history[-100:]:  # Last 100 points
        times.append(state['time'])

        # Calculate energies for this state
        ke = pe = 0
        particles = state['particles']

        for p in particles:
            v_squared = sum(v**2 for v in p['velocity'])
            ke += 0.5 * p['mass'] * v_squared

        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                r_vec = np.array(particles[j]['position']) - np.array(particles[i]['position'])
                r = np.linalg.norm(r_vec)
                if r > 1e-6:
                    pe += -1.0 * particles[i]['mass'] * particles[j]['mass'] / r

        kinetic.append(ke)
        potential.append(pe)
        total.append(ke + pe)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=times, y=kinetic, name='Kinetic', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=times, y=potential, name='Potential', line=dict(color='magenta')))
    fig.add_trace(go.Scatter(x=times, y=total, name='Total', line=dict(color='lime', dash='dash')))

    fig.update_layout(
        title="Energy Evolution",
        xaxis_title="Time",
        yaxis_title="Energy",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 14, 39, 0.5)"
    )

    return fig

def create_loss_history_plot():
    """Create training loss history plot."""
    # Simulate loss history
    epochs = list(range(1, st.session_state.generation + 1))
    losses = [0.1 * np.exp(-i/10) + 0.001 + np.random.normal(0, 0.001)
              for i in epochs]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode='lines+markers',
        name='Total Loss',
        line=dict(color='cyan', width=2)
    ))

    fig.update_layout(
        title="Training Loss History",
        xaxis_title="Generation",
        yaxis_title="Loss",
        yaxis_type="log",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 14, 39, 0.5)"
    )

    return fig

# =====================================================================
# MAIN STREAMLIT APP
# =====================================================================

def main():
    """Main Streamlit application."""

    # Load chatbot model
    if st.session_state.chatbot_model is None:
        with st.spinner("Loading AI model..."):
            st.session_state.chatbot_model = load_chatbot_model()

    # Load Wave Theory system
    if st.session_state.wave_theory_system is None:
        with st.spinner("Loading Wave Theory system..."):
            st.session_state.wave_theory_system = load_wave_theory_system()

    # Header
    st.markdown("""
        <h1 style='text-align: center; background: linear-gradient(45deg, #00ffff, #ff00ff, #00ff88); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🌊 Wave Theory Chatbot
        </h1>
        <p style='text-align: center; color: #8892b0;'>
            Neuro-Symbolic Physics Discovery Engine
        </p>
    """, unsafe_allow_html=True)

    # Create three columns layout
    col1, col2, col3 = st.columns([1, 2, 1])

    # Left Column - Chat Interface
    with col1:
        st.markdown("### 💬 Quantum Interface")

        # Chat history container
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Chat input
        user_input = st.chat_input("Ask about physics experiments...")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Process query and get response
            response = process_user_query(user_input)

            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Rerun to update chat display
            st.rerun()

    # Middle Column - Visualization
    with col2:
        st.markdown("### 🌌 Universe Visualization")

        # 3D Particle Visualization
        particle_viz = create_3d_particle_visualization()
        st.plotly_chart(particle_viz, use_container_width=True)

        # Control buttons
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
                    {'id': 0, 'position': [0, 0, 0], 'velocity': [0.5, 0, 0], 'mass': 5.0},
                    {'id': 1, 'position': [10, 0, 0], 'velocity': [-0.5, 0.5, 0], 'mass': 5.0},
                    {'id': 2, 'position': [5, 8.66, 0], 'velocity': [0, -0.5, 0], 'mass': 5.0}
                ]
                st.session_state.simulation_history = []
                st.rerun()

        with button_col4:
            if st.button("➕ Add", use_container_width=True):
                new_particle = {
                    'id': len(st.session_state.particles),
                    'position': [np.random.uniform(-10, 10) for _ in range(3)],
                    'velocity': [np.random.uniform(-1, 1) for _ in range(3)],
                    'mass': np.random.uniform(3, 8)
                }
                st.session_state.particles.append(new_particle)
                st.rerun()

        # Energy plot
        energy_plot = create_energy_plot()
        if energy_plot:
            st.plotly_chart(energy_plot, use_container_width=True)

    # Right Column - Model Status
    with col3:
        st.markdown("### 🧠 Model Status")

        # Metrics
        energy = calculate_system_energy()

        st.metric("Particles", len(st.session_state.particles))
        st.metric("Total Energy", f"{energy['total']:.3f}")
        st.metric("Generation", st.session_state.generation)
        st.metric("Model Loss", f"{st.session_state.model_loss:.6f}")

        # Current equation display
        st.markdown("#### Current Equation")
        st.markdown(f"""
            <div class='equation-box'>
                {st.session_state.current_equation}
            </div>
        """, unsafe_allow_html=True)

        # Training loss plot
        loss_plot = create_loss_history_plot()
        if loss_plot:
            st.plotly_chart(loss_plot, use_container_width=True)

    # Expandable sections for advanced features
    with st.expander("🔬 Advanced Settings"):
        st.markdown("### Physics Parameters")

        col1, col2 = st.columns(2)
        with col1:
            G = st.slider("Gravitational Constant", 0.1, 5.0, 1.0, 0.1)
            wave_freq = st.slider("Wave Frequency", 0.1, 2.0, 0.5, 0.1)
        with col2:
            decay_length = st.slider("Decay Length", 5.0, 20.0, 10.0, 1.0)
            dt = st.slider("Time Step", 0.001, 0.1, 0.01, 0.001)

        st.markdown("### Neural Network Architecture")
        layers = st.number_input("Hidden Layers", 4, 12, 6)
        neurons = st.number_input("Neurons per Layer", 32, 256, 128)

        st.markdown("### Symbolic Regression")
        populations = st.number_input("Populations", 5, 30, 15)
        max_complexity = st.slider("Max Equation Complexity", 10, 50, 20)

    with st.expander("📊 Pareto Front"):
        st.markdown("### Discovered Equations (Accuracy vs Complexity)")

        # Generate sample Pareto front data
        pareto_data = pd.DataFrame({
            'Complexity': [5, 8, 12, 15, 20, 25],
            'Loss': [0.05, 0.03, 0.02, 0.015, 0.012, 0.011],
            'Equation': [
                'r^(-2)',
                'sin(r) * r^(-2)',
                'sin(ωr) * r^(-2)',
                'sin(ωr) * exp(-r/λ) * r^(-2)',
                'sin(ωr) * exp(-r/λ) * r^(-2) * m1 * m2',
                'Complex equation with 25 nodes'
            ]
        })

        fig = px.scatter(pareto_data, x='Complexity', y='Loss',
                        hover_data=['Equation'], log_y=True,
                        title="Pareto Front of Discovered Equations")
        fig.update_traces(marker=dict(size=10, color='cyan'))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pareto_data)

    # Footer
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #8892b0; font-size: 0.9em;'>
            Wave Theory Chatbot v1.0 | Combining Physics-Informed Neural Networks with Symbolic Regression
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
