"""
Configuration loader for Wave Theory system
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to configs/config.yaml
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}

def get_physics_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract physics configuration."""
    return config.get('physics', {})

def get_neural_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract neural network configuration."""
    return config.get('neural_network', {})

def get_symbolic_regression_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract symbolic regression configuration."""
    return config.get('symbolic_regression', {})

def get_chatbot_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract chatbot configuration."""
    return config.get('chatbot', {})

def get_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract logging configuration."""
    return config.get('logging', {})
