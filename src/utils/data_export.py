"""
Export/Import Functionality for Simulations
Comprehensive data export and import system
"""

import numpy as np
import pandas as pd
import json
import pickle
import h5py
import yaml
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import zipfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimulationData:
    """Complete simulation data structure."""
    metadata: Dict[str, Any]
    particles: List[Dict[str, Any]]
    physics_params: Dict[str, float]
    simulation_params: Dict[str, Any]
    time_series: List[Dict[str, Any]]
    model_state: Optional[Dict[str, Any]] = None
    analytics: Optional[Dict[str, Any]] = None

class SimulationExporter:
    """Export simulation data in various formats."""
    
    def __init__(self, output_dir: str = "data/exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_simulation(self, simulation_data: SimulationData, 
                         format: str = "json", filename: Optional[str] = None) -> str:
        """Export simulation data in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"simulation_{timestamp}"
        
        filepath = self.output_dir / f"{filename}.{format}"
        
        if format == "json":
            return self._export_json(simulation_data, filepath)
        elif format == "csv":
            return self._export_csv(simulation_data, filepath)
        elif format == "hdf5":
            return self._export_hdf5(simulation_data, filepath)
        elif format == "pickle":
            return self._export_pickle(simulation_data, filepath)
        elif format == "yaml":
            return self._export_yaml(simulation_data, filepath)
        elif format == "zip":
            return self._export_zip(simulation_data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, data: SimulationData, filepath: Path) -> str:
        """Export to JSON format."""
        export_data = {
            'metadata': data.metadata,
            'particles': data.particles,
            'physics_params': data.physics_params,
            'simulation_params': data.simulation_params,
            'time_series': data.time_series,
            'model_state': data.model_state,
            'analytics': data.analytics,
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'format': 'json',
                'version': '1.0'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported simulation to JSON: {filepath}")
        return str(filepath)
    
    def _export_csv(self, data: SimulationData, filepath: Path) -> str:
        """Export to CSV format."""
        # Create a comprehensive CSV with all data
        csv_data = []
        
        # Add metadata
        csv_data.append(['# Simulation Metadata'])
        for key, value in data.metadata.items():
            csv_data.append([f"# {key}", str(value)])
        
        csv_data.append([])  # Empty row
        
        # Add physics parameters
        csv_data.append(['# Physics Parameters'])
        for key, value in data.physics_params.items():
            csv_data.append([f"# {key}", str(value)])
        
        csv_data.append([])  # Empty row
        
        # Add simulation parameters
        csv_data.append(['# Simulation Parameters'])
        for key, value in data.simulation_params.items():
            csv_data.append([f"# {key}", str(value)])
        
        csv_data.append([])  # Empty row
        
        # Add particle data
        if data.particles:
            csv_data.append(['# Particle Data'])
            particle_df = pd.DataFrame(data.particles)
            csv_data.extend(particle_df.values.tolist())
        
        csv_data.append([])  # Empty row
        
        # Add time series data
        if data.time_series:
            csv_data.append(['# Time Series Data'])
            time_df = pd.DataFrame(data.time_series)
            csv_data.extend(time_df.values.tolist())
        
        # Write to CSV
        with open(filepath, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(csv_data)
        
        logger.info(f"Exported simulation to CSV: {filepath}")
        return str(filepath)
    
    def _export_hdf5(self, data: SimulationData, filepath: Path) -> str:
        """Export to HDF5 format for efficient storage."""
        with h5py.File(filepath, 'w') as f:
            # Create groups
            metadata_group = f.create_group('metadata')
            particles_group = f.create_group('particles')
            physics_group = f.create_group('physics')
            simulation_group = f.create_group('simulation')
            time_series_group = f.create_group('time_series')
            
            # Store metadata
            for key, value in data.metadata.items():
                metadata_group.attrs[key] = str(value)
            
            # Store physics parameters
            for key, value in data.physics_params.items():
                physics_group.attrs[key] = value
            
            # Store simulation parameters
            for key, value in data.simulation_params.items():
                simulation_group.attrs[key] = value
            
            # Store particle data
            if data.particles:
                particle_df = pd.DataFrame(data.particles)
                for col in particle_df.columns:
                    particles_group.create_dataset(col, data=particle_df[col].values)
            
            # Store time series data
            if data.time_series:
                time_df = pd.DataFrame(data.time_series)
                for col in time_df.columns:
                    time_series_group.create_dataset(col, data=time_df[col].values)
            
            # Store model state if available
            if data.model_state:
                model_group = f.create_group('model_state')
                for key, value in data.model_state.items():
                    if isinstance(value, np.ndarray):
                        model_group.create_dataset(key, data=value)
                    else:
                        model_group.attrs[key] = str(value)
            
            # Store analytics if available
            if data.analytics:
                analytics_group = f.create_group('analytics')
                for key, value in data.analytics.items():
                    if isinstance(value, (list, np.ndarray)):
                        analytics_group.create_dataset(key, data=np.array(value))
                    else:
                        analytics_group.attrs[key] = value
        
        logger.info(f"Exported simulation to HDF5: {filepath}")
        return str(filepath)
    
    def _export_pickle(self, data: SimulationData, filepath: Path) -> str:
        """Export to pickle format for Python objects."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Exported simulation to pickle: {filepath}")
        return str(filepath)
    
    def _export_yaml(self, data: SimulationData, filepath: Path) -> str:
        """Export to YAML format."""
        export_data = {
            'metadata': data.metadata,
            'particles': data.particles,
            'physics_params': data.physics_params,
            'simulation_params': data.simulation_params,
            'time_series': data.time_series,
            'model_state': data.model_state,
            'analytics': data.analytics,
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'format': 'yaml',
                'version': '1.0'
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False)
        
        logger.info(f"Exported simulation to YAML: {filepath}")
        return str(filepath)
    
    def _export_zip(self, data: SimulationData, filepath: Path) -> str:
        """Export to ZIP format with multiple files."""
        zip_path = filepath.with_suffix('.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Export main data as JSON
            json_data = {
                'metadata': data.metadata,
                'particles': data.particles,
                'physics_params': data.physics_params,
                'simulation_params': data.simulation_params,
                'time_series': data.time_series,
                'model_state': data.model_state,
                'analytics': data.analytics
            }
            
            zipf.writestr('simulation_data.json', json.dumps(json_data, indent=2, default=str))
            
            # Export particle data as CSV
            if data.particles:
                particle_df = pd.DataFrame(data.particles)
                zipf.writestr('particles.csv', particle_df.to_csv(index=False))
            
            # Export time series as CSV
            if data.time_series:
                time_df = pd.DataFrame(data.time_series)
                zipf.writestr('time_series.csv', time_df.to_csv(index=False))
            
            # Export parameters as YAML
            params_data = {
                'physics_params': data.physics_params,
                'simulation_params': data.simulation_params
            }
            zipf.writestr('parameters.yaml', yaml.dump(params_data, default_flow_style=False))
            
            # Add README
            readme_content = f"""
# Wave Theory Simulation Export

## Export Information
- Timestamp: {datetime.now().isoformat()}
- Format: ZIP
- Version: 1.0

## Files Included
- simulation_data.json: Complete simulation data
- particles.csv: Particle data in CSV format
- time_series.csv: Time series data in CSV format
- parameters.yaml: Simulation parameters

## Usage
This export contains all data from a Wave Theory simulation run.
You can import this data back into the system or analyze it externally.
"""
            zipf.writestr('README.md', readme_content)
        
        logger.info(f"Exported simulation to ZIP: {zip_path}")
        return str(zip_path)
    
    def export_analytics_only(self, analytics_data: Dict[str, Any], 
                             format: str = "json") -> str:
        """Export only analytics data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)
        elif format == "csv":
            # Convert analytics to CSV format
            df = pd.DataFrame([analytics_data])
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format for analytics: {format}")
        
        logger.info(f"Exported analytics to {format}: {filepath}")
        return str(filepath)

class SimulationImporter:
    """Import simulation data from various formats."""
    
    def __init__(self, input_dir: str = "data/exports"):
        self.input_dir = Path(input_dir)
    
    def import_simulation(self, filepath: str) -> Optional[SimulationData]:
        """Import simulation data from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
        
        format = filepath.suffix[1:]  # Remove the dot
        
        try:
            if format == "json":
                return self._import_json(filepath)
            elif format == "csv":
                return self._import_csv(filepath)
            elif format == "h5" or format == "hdf5":
                return self._import_hdf5(filepath)
            elif format == "pickle":
                return self._import_pickle(filepath)
            elif format == "yaml" or format == "yml":
                return self._import_yaml(filepath)
            elif format == "zip":
                return self._import_zip(filepath)
            else:
                logger.error(f"Unsupported format: {format}")
                return None
        except Exception as e:
            logger.error(f"Error importing simulation: {e}")
            return None
    
    def _import_json(self, filepath: Path) -> SimulationData:
        """Import from JSON format."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return SimulationData(
            metadata=data.get('metadata', {}),
            particles=data.get('particles', []),
            physics_params=data.get('physics_params', {}),
            simulation_params=data.get('simulation_params', {}),
            time_series=data.get('time_series', []),
            model_state=data.get('model_state'),
            analytics=data.get('analytics')
        )
    
    def _import_csv(self, filepath: Path) -> SimulationData:
        """Import from CSV format."""
        # This is a simplified CSV import
        # In practice, you'd need to parse the CSV structure more carefully
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse metadata, parameters, and data
        metadata = {}
        physics_params = {}
        simulation_params = {}
        particles = []
        time_series = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                if 'Metadata' in line:
                    current_section = 'metadata'
                elif 'Physics' in line:
                    current_section = 'physics'
                elif 'Simulation' in line:
                    current_section = 'simulation'
                elif 'Particle' in line:
                    current_section = 'particles'
                elif 'Time Series' in line:
                    current_section = 'time_series'
            elif line and not line.startswith('#'):
                if current_section == 'metadata':
                    parts = line.split(',')
                    if len(parts) >= 2:
                        metadata[parts[0]] = parts[1]
                elif current_section == 'physics':
                    parts = line.split(',')
                    if len(parts) >= 2:
                        physics_params[parts[0]] = float(parts[1])
                elif current_section == 'simulation':
                    parts = line.split(',')
                    if len(parts) >= 2:
                        simulation_params[parts[0]] = parts[1]
                # Add more parsing for particles and time series as needed
        
        return SimulationData(
            metadata=metadata,
            particles=particles,
            physics_params=physics_params,
            simulation_params=simulation_params,
            time_series=time_series
        )
    
    def _import_hdf5(self, filepath: Path) -> SimulationData:
        """Import from HDF5 format."""
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            metadata = dict(f['metadata'].attrs)
            
            # Load physics parameters
            physics_params = dict(f['physics'].attrs)
            
            # Load simulation parameters
            simulation_params = dict(f['simulation'].attrs)
            
            # Load particle data
            particles = []
            if 'particles' in f:
                particle_data = {}
                for key in f['particles'].keys():
                    particle_data[key] = f['particles'][key][:]
                
                if particle_data:
                    particle_df = pd.DataFrame(particle_data)
                    particles = particle_df.to_dict('records')
            
            # Load time series data
            time_series = []
            if 'time_series' in f:
                time_data = {}
                for key in f['time_series'].keys():
                    time_data[key] = f['time_series'][key][:]
                
                if time_data:
                    time_df = pd.DataFrame(time_data)
                    time_series = time_df.to_dict('records')
            
            # Load model state
            model_state = None
            if 'model_state' in f:
                model_state = {}
                for key in f['model_state'].keys():
                    model_state[key] = f['model_state'][key][:]
                for key in f['model_state'].attrs.keys():
                    model_state[key] = f['model_state'].attrs[key]
            
            # Load analytics
            analytics = None
            if 'analytics' in f:
                analytics = {}
                for key in f['analytics'].keys():
                    analytics[key] = f['analytics'][key][:]
                for key in f['analytics'].attrs.keys():
                    analytics[key] = f['analytics'].attrs[key]
        
        return SimulationData(
            metadata=metadata,
            particles=particles,
            physics_params=physics_params,
            simulation_params=simulation_params,
            time_series=time_series,
            model_state=model_state,
            analytics=analytics
        )
    
    def _import_pickle(self, filepath: Path) -> SimulationData:
        """Import from pickle format."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _import_yaml(self, filepath: Path) -> SimulationData:
        """Import from YAML format."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return SimulationData(
            metadata=data.get('metadata', {}),
            particles=data.get('particles', []),
            physics_params=data.get('physics_params', {}),
            simulation_params=data.get('simulation_params', {}),
            time_series=data.get('time_series', []),
            model_state=data.get('model_state'),
            analytics=data.get('analytics')
        )
    
    def _import_zip(self, filepath: Path) -> SimulationData:
        """Import from ZIP format."""
        with zipfile.ZipFile(filepath, 'r') as zipf:
            # Read main simulation data
            json_data = json.loads(zipf.read('simulation_data.json'))
            
            return SimulationData(
                metadata=json_data.get('metadata', {}),
                particles=json_data.get('particles', []),
                physics_params=json_data.get('physics_params', {}),
                simulation_params=json_data.get('simulation_params', {}),
                time_series=json_data.get('time_series', []),
                model_state=json_data.get('model_state'),
                analytics=json_data.get('analytics')
            )
    
    def list_available_exports(self) -> List[Dict[str, Any]]:
        """List all available export files."""
        exports = []
        
        for file_path in self.input_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                exports.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'format': file_path.suffix[1:]
                })
        
        return sorted(exports, key=lambda x: x['modified'], reverse=True)

# Global instances
simulation_exporter = SimulationExporter()
simulation_importer = SimulationImporter()
