"""
Model Performance Monitoring and Visualization
Real-time monitoring of PINN and symbolic regression performance
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Metrics for model performance monitoring."""
    timestamp: str
    model_type: str  # 'pinn' or 'symbolic'
    generation: int
    loss: float
    accuracy: float
    convergence_rate: float
    equation_complexity: int
    physics_residual: float
    training_time: float
    inference_time: float
    memory_usage: float
    gpu_usage: Optional[float] = None

class ModelPerformanceMonitor:
    """Real-time model performance monitoring system."""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.thresholds = {
            'loss': 0.01,
            'accuracy': 0.95,
            'convergence_rate': 0.001,
            'training_time': 60.0,  # seconds
            'memory_usage': 0.8  # 80%
        }
    
    def log_metrics(self, metrics: ModelMetrics) -> None:
        """Log model performance metrics."""
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Keep only recent history (last 1000 entries)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _check_alerts(self, metrics: ModelMetrics) -> None:
        """Check for performance alerts."""
        alerts = []
        
        if metrics.loss > self.thresholds['loss']:
            alerts.append(f"High loss detected: {metrics.loss:.4f}")
        
        if metrics.accuracy < self.thresholds['accuracy']:
            alerts.append(f"Low accuracy: {metrics.accuracy:.4f}")
        
        if metrics.training_time > self.thresholds['training_time']:
            alerts.append(f"Slow training: {metrics.training_time:.2f}s")
        
        if metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1%}")
        
        for alert in alerts:
            self.alerts.append({
                'timestamp': metrics.timestamp,
                'model_type': metrics.model_type,
                'message': alert,
                'severity': 'warning'
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        
        return {
            'total_metrics': len(self.metrics_history),
            'avg_loss': df['loss'].mean(),
            'min_loss': df['loss'].min(),
            'avg_accuracy': df['accuracy'].mean(),
            'max_accuracy': df['accuracy'].max(),
            'avg_training_time': df['training_time'].mean(),
            'total_training_time': df['training_time'].sum(),
            'convergence_rate': df['convergence_rate'].iloc[-1] if len(df) > 0 else 0,
            'active_alerts': len([a for a in self.alerts if a['severity'] == 'warning']),
            'last_update': self.metrics_history[-1].timestamp if self.metrics_history else None
        }
    
    def create_performance_dashboard(self) -> Dict[str, go.Figure]:
        """Create comprehensive performance monitoring dashboard."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        dashboard = {}
        
        # Loss Evolution
        fig_loss = go.Figure()
        
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            fig_loss.add_trace(go.Scatter(
                x=model_data['timestamp'],
                y=model_data['loss'],
                mode='lines+markers',
                name=f'{model_type.upper()} Loss',
                line=dict(width=2)
            ))
        
        fig_loss.update_layout(
            title="Model Loss Evolution",
            xaxis_title="Time",
            yaxis_title="Loss",
            height=400,
            yaxis=dict(type="log")
        )
        dashboard['loss_evolution'] = fig_loss
        
        # Accuracy Trends
        fig_accuracy = go.Figure()
        
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            fig_accuracy.add_trace(go.Scatter(
                x=model_data['timestamp'],
                y=model_data['accuracy'],
                mode='lines+markers',
                name=f'{model_type.upper()} Accuracy',
                line=dict(width=2)
            ))
        
        fig_accuracy.update_layout(
            title="Model Accuracy Trends",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            height=400
        )
        dashboard['accuracy_trends'] = fig_accuracy
        
        # Training Performance
        fig_training = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Time', 'Memory Usage', 'Convergence Rate', 'Physics Residual'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            
            fig_training.add_trace(
                go.Scatter(x=model_data['timestamp'], y=model_data['training_time'],
                          name=f'{model_type} Training Time', line=dict(width=2)),
                row=1, col=1
            )
            fig_training.add_trace(
                go.Scatter(x=model_data['timestamp'], y=model_data['memory_usage'],
                          name=f'{model_type} Memory', line=dict(width=2)),
                row=1, col=2
            )
            fig_training.add_trace(
                go.Scatter(x=model_data['timestamp'], y=model_data['convergence_rate'],
                          name=f'{model_type} Convergence', line=dict(width=2)),
                row=2, col=1
            )
            fig_training.add_trace(
                go.Scatter(x=model_data['timestamp'], y=model_data['physics_residual'],
                          name=f'{model_type} Physics', line=dict(width=2)),
                row=2, col=2
            )
        
        fig_training.update_layout(height=600, title_text="Training Performance Metrics")
        dashboard['training_performance'] = fig_training
        
        # Model Comparison
        fig_comparison = go.Figure()
        
        pinn_data = df[df['model_type'] == 'pinn']
        symbolic_data = df[df['model_type'] == 'symbolic']
        
        if not pinn_data.empty and not symbolic_data.empty:
            fig_comparison.add_trace(go.Scatter(
                x=pinn_data['generation'],
                y=pinn_data['loss'],
                mode='lines+markers',
                name='PINN Loss',
                line=dict(color='#ff6b6b', width=3)
            ))
            fig_comparison.add_trace(go.Scatter(
                x=symbolic_data['generation'],
                y=symbolic_data['loss'],
                mode='lines+markers',
                name='Symbolic Loss',
                line=dict(color='#4ecdc4', width=3)
            ))
        
        fig_comparison.update_layout(
            title="PINN vs Symbolic Regression Performance",
            xaxis_title="Generation",
            yaxis_title="Loss",
            height=400
        )
        dashboard['model_comparison'] = fig_comparison
        
        # Equation Complexity Evolution
        fig_complexity = go.Figure()
        
        symbolic_data = df[df['model_type'] == 'symbolic']
        if not symbolic_data.empty:
            fig_complexity.add_trace(go.Scatter(
                x=symbolic_data['timestamp'],
                y=symbolic_data['equation_complexity'],
                mode='lines+markers',
                name='Equation Complexity',
                line=dict(color='#96ceb4', width=2)
            ))
        
        fig_complexity.update_layout(
            title="Symbolic Equation Complexity Evolution",
            xaxis_title="Time",
            yaxis_title="Complexity",
            height=400
        )
        dashboard['equation_complexity'] = fig_complexity
        
        return dashboard
    
    def create_alerts_panel(self) -> Dict[str, Any]:
        """Create alerts and notifications panel."""
        recent_alerts = [a for a in self.alerts if 
                        datetime.fromisoformat(a['timestamp']) > 
                        datetime.now() - timedelta(hours=24)]
        
        return {
            'total_alerts': len(self.alerts),
            'recent_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a['severity'] == 'critical']),
            'warning_alerts': len([a for a in recent_alerts if a['severity'] == 'warning']),
            'alerts': recent_alerts[-10:]  # Last 10 alerts
        }
    
    def export_performance_report(self, format: str = 'json') -> str:
        """Export performance report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_performance_report_{timestamp}.{format}"
        
        report_data = {
            'summary': self.get_performance_summary(),
            'alerts': self.alerts,
            'metrics': [asdict(m) for m in self.metrics_history[-100:]]  # Last 100 metrics
        }
        
        if format == 'json':
            with open(f"data/reports/{filename}", "w") as f:
                json.dump(report_data, f, indent=2, default=str)
        elif format == 'csv':
            df = pd.DataFrame([asdict(m) for m in self.metrics_history])
            df.to_csv(f"data/reports/{filename}", index=False)
        
        return filename

class RealTimeMonitor:
    """Real-time monitoring with live updates."""
    
    def __init__(self, monitor: ModelPerformanceMonitor):
        self.monitor = monitor
        self.is_monitoring = False
        self.update_interval = 1.0  # seconds
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        self.is_monitoring = True
        logger.info("Started real-time model monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_monitoring = False
        logger.info("Stopped real-time model monitoring")
    
    def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live metrics."""
        if not self.monitor.metrics_history:
            return {}
        
        latest = self.monitor.metrics_history[-1]
        return {
            'timestamp': latest.timestamp,
            'model_type': latest.model_type,
            'loss': latest.loss,
            'accuracy': latest.accuracy,
            'convergence_rate': latest.convergence_rate,
            'training_time': latest.training_time,
            'memory_usage': latest.memory_usage
        }

# Global monitoring instances
model_monitor = ModelPerformanceMonitor()
realtime_monitor = RealTimeMonitor(model_monitor)
