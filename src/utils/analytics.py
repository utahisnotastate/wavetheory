"""
Real-time Analytics and Data Logging System
Comprehensive monitoring and analysis for Wave Theory experiments
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run."""
    timestamp: str
    experiment_id: str
    generation: int
    model_loss: float
    physics_loss: float
    total_energy: float
    kinetic_energy: float
    potential_energy: float
    particle_count: int
    simulation_time: float
    equation_complexity: int
    equation_accuracy: float
    convergence_rate: float
    user_queries: int
    simulation_steps: int

@dataclass
class SystemPerformance:
    """System performance metrics."""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    model_inference_time: float
    simulation_step_time: float
    ui_response_time: float

class AnalyticsEngine:
    """Real-time analytics and data logging engine."""
    
    def __init__(self, db_path: str = "data/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Real-time data storage
        self.current_experiment = None
        self.metrics_history = []
        self.performance_history = []
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    experiment_id TEXT,
                    generation INTEGER,
                    model_loss REAL,
                    physics_loss REAL,
                    total_energy REAL,
                    kinetic_energy REAL,
                    potential_energy REAL,
                    particle_count INTEGER,
                    simulation_time REAL,
                    equation_complexity INTEGER,
                    equation_accuracy REAL,
                    convergence_rate REAL,
                    user_queries INTEGER,
                    simulation_steps INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    model_inference_time REAL,
                    simulation_step_time REAL,
                    ui_response_time REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_id TEXT,
                    query_type TEXT,
                    query_text TEXT,
                    response_time REAL,
                    success BOOLEAN
                )
            """)
    
    def start_experiment(self, experiment_id: str) -> None:
        """Start tracking a new experiment."""
        self.current_experiment = {
            'id': experiment_id,
            'start_time': datetime.now(),
            'queries': 0,
            'steps': 0
        }
        logger.info(f"Started experiment: {experiment_id}")
    
    def log_metrics(self, metrics: ExperimentMetrics) -> None:
        """Log experiment metrics."""
        if self.current_experiment:
            metrics.experiment_id = self.current_experiment['id']
            self.current_experiment['steps'] += 1
            
        # Store in memory
        self.metrics_history.append(metrics)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiment_metrics 
                (timestamp, experiment_id, generation, model_loss, physics_loss,
                 total_energy, kinetic_energy, potential_energy, particle_count,
                 simulation_time, equation_complexity, equation_accuracy,
                 convergence_rate, user_queries, simulation_steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.experiment_id, metrics.generation,
                metrics.model_loss, metrics.physics_loss, metrics.total_energy,
                metrics.kinetic_energy, metrics.potential_energy, metrics.particle_count,
                metrics.simulation_time, metrics.equation_complexity,
                metrics.equation_accuracy, metrics.convergence_rate,
                metrics.user_queries, metrics.simulation_steps
            ))
    
    def log_performance(self, performance: SystemPerformance) -> None:
        """Log system performance metrics."""
        self.performance_history.append(performance)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_performance 
                (timestamp, cpu_usage, memory_usage, gpu_usage,
                 model_inference_time, simulation_step_time, ui_response_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                performance.timestamp, performance.cpu_usage, performance.memory_usage,
                performance.gpu_usage, performance.model_inference_time,
                performance.simulation_step_time, performance.ui_response_time
            ))
    
    def log_user_interaction(self, session_id: str, query_type: str, 
                           query_text: str, response_time: float, success: bool) -> None:
        """Log user interaction."""
        if self.current_experiment:
            self.current_experiment['queries'] += 1
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_interactions 
                (timestamp, session_id, query_type, query_text, response_time, success)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(), session_id, query_type,
                query_text, response_time, success
            ))
    
    def get_experiment_summary(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            if experiment_id:
                query = "SELECT * FROM experiment_metrics WHERE experiment_id = ?"
                params = (experiment_id,)
            else:
                query = "SELECT * FROM experiment_metrics ORDER BY timestamp DESC LIMIT 100"
                params = ()
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return {}
            
            return {
                'total_experiments': df['experiment_id'].nunique(),
                'avg_model_loss': df['model_loss'].mean(),
                'min_model_loss': df['model_loss'].min(),
                'avg_energy': df['total_energy'].mean(),
                'avg_particles': df['particle_count'].mean(),
                'total_generations': df['generation'].max(),
                'convergence_rate': df['convergence_rate'].mean(),
                'total_queries': df['user_queries'].sum(),
                'total_steps': df['simulation_steps'].sum()
            }
    
    def create_analytics_dashboard(self) -> Dict[str, go.Figure]:
        """Create comprehensive analytics dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            # Get recent data
            df_metrics = pd.read_sql_query("""
                SELECT * FROM experiment_metrics 
                WHERE timestamp >= datetime('now', '-24 hours')
                ORDER BY timestamp
            """, conn)
            
            df_performance = pd.read_sql_query("""
                SELECT * FROM system_performance 
                WHERE timestamp >= datetime('now', '-1 hour')
                ORDER BY timestamp
            """, conn)
        
        dashboard = {}
        
        if not df_metrics.empty:
            # Model Loss Over Time
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=df_metrics['timestamp'], 
                y=df_metrics['model_loss'],
                mode='lines+markers',
                name='Model Loss',
                line=dict(color='#ff6b6b', width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                x=df_metrics['timestamp'], 
                y=df_metrics['physics_loss'],
                mode='lines+markers',
                name='Physics Loss',
                line=dict(color='#4ecdc4', width=2)
            ))
            fig_loss.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Time",
                yaxis_title="Loss",
                height=400
            )
            dashboard['loss_evolution'] = fig_loss
            
            # Energy Evolution
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                x=df_metrics['timestamp'], 
                y=df_metrics['total_energy'],
                mode='lines+markers',
                name='Total Energy',
                line=dict(color='#45b7d1', width=2)
            ))
            fig_energy.add_trace(go.Scatter(
                x=df_metrics['timestamp'], 
                y=df_metrics['kinetic_energy'],
                mode='lines+markers',
                name='Kinetic Energy',
                line=dict(color='#96ceb4', width=2)
            ))
            fig_energy.add_trace(go.Scatter(
                x=df_metrics['timestamp'], 
                y=df_metrics['potential_energy'],
                mode='lines+markers',
                name='Potential Energy',
                line=dict(color='#feca57', width=2)
            ))
            fig_energy.update_layout(
                title="Energy Evolution",
                xaxis_title="Time",
                yaxis_title="Energy",
                height=400
            )
            dashboard['energy_evolution'] = fig_energy
            
            # Convergence Analysis
            fig_convergence = go.Figure()
            fig_convergence.add_trace(go.Scatter(
                x=df_metrics['generation'], 
                y=df_metrics['convergence_rate'],
                mode='lines+markers',
                name='Convergence Rate',
                line=dict(color='#ff9ff3', width=2)
            ))
            fig_convergence.update_layout(
                title="Convergence Rate by Generation",
                xaxis_title="Generation",
                yaxis_title="Convergence Rate",
                height=400
            )
            dashboard['convergence'] = fig_convergence
        
        if not df_performance.empty:
            # System Performance
            fig_perf = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Model Inference Time', 'Simulation Step Time'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_perf.add_trace(
                go.Scatter(x=df_performance['timestamp'], y=df_performance['cpu_usage'], 
                          name='CPU %', line=dict(color='#ff6b6b')),
                row=1, col=1
            )
            fig_perf.add_trace(
                go.Scatter(x=df_performance['timestamp'], y=df_performance['memory_usage'], 
                          name='Memory %', line=dict(color='#4ecdc4')),
                row=1, col=2
            )
            fig_perf.add_trace(
                go.Scatter(x=df_performance['timestamp'], y=df_performance['model_inference_time'], 
                          name='Inference Time (ms)', line=dict(color='#45b7d1')),
                row=2, col=1
            )
            fig_perf.add_trace(
                go.Scatter(x=df_performance['timestamp'], y=df_performance['simulation_step_time'], 
                          name='Step Time (ms)', line=dict(color='#96ceb4')),
                row=2, col=2
            )
            
            fig_perf.update_layout(height=600, title_text="System Performance Metrics")
            dashboard['performance'] = fig_perf
        
        return dashboard
    
    def export_data(self, format: str = 'csv', experiment_id: Optional[str] = None) -> str:
        """Export analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            if experiment_id:
                df = pd.read_sql_query(
                    "SELECT * FROM experiment_metrics WHERE experiment_id = ?", 
                    conn, params=(experiment_id,)
                )
            else:
                df = pd.read_sql_query("SELECT * FROM experiment_metrics", conn)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wave_theory_analytics_{timestamp}.{format}"
        filepath = Path("data/exports") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
        
        return str(filepath)

# Global analytics instance
analytics = AnalyticsEngine()
