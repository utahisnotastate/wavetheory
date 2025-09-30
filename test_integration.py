#!/usr/bin/env python3
"""
Integration test for Wave Theory system
Tests that all components can be imported and initialized
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.neuro_symbolic import WaveTheorySystem, WaveTheoryConfig, create_wave_theory_system
        print("✅ Neuro-symbolic module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import neuro-symbolic module: {e}")
        return False
    
    try:
        from models.pinn_jax import WavePINN, PINNTrainer, create_pinn_model
        print("✅ PINN module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import PINN module: {e}")
        return False
    
    try:
        from models.symbolic_engine import SymbolicRegressionEngine, PySRConfig
        print("✅ Symbolic engine module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import symbolic engine module: {e}")
        return False
    
    try:
        from utils.config_loader import load_config
        print("✅ Config loader imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config loader: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from utils.config_loader import load_config
        config = load_config()
        
        if config:
            print("✅ Configuration loaded successfully")
            print(f"   - App name: {config.get('app', {}).get('name', 'Unknown')}")
            print(f"   - Physics G: {config.get('physics', {}).get('force_law', {}).get('G', 'Unknown')}")
            return True
        else:
            print("❌ Configuration is empty")
            return False
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def test_wave_theory_system():
    """Test Wave Theory system initialization."""
    print("\nTesting Wave Theory system initialization...")
    
    try:
        from models.neuro_symbolic import create_wave_theory_system
        import jax
        
        key = jax.random.PRNGKey(42)
        system = create_wave_theory_system(key=key)
        
        print("✅ Wave Theory system created successfully")
        print(f"   - Current equation: {system.get_current_equation()}")
        print(f"   - Generation: {system.get_generation()}")
        return True
    except Exception as e:
        print(f"❌ Failed to create Wave Theory system: {e}")
        return False

def test_simulation():
    """Test simulation functionality."""
    print("\nTesting simulation...")
    
    try:
        from models.neuro_symbolic import create_wave_theory_system
        import jax
        
        key = jax.random.PRNGKey(42)
        system = create_wave_theory_system(key=key)
        
        # Run a short simulation
        sim_data = system.run_simulation(steps=10, save_interval=5)
        
        print("✅ Simulation completed successfully")
        print(f"   - Generated {len(sim_data)} data points")
        print(f"   - First data point time: {sim_data[0]['time']}")
        print(f"   - Number of particles: {len(sim_data[0]['bodies'])}")
        return True
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Wave Theory Integration Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config_loading,
        test_wave_theory_system,
        test_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
