#!/usr/bin/env python3
"""
Wave Theory Chatbot - Application Runner
Simple script to run the Streamlit application
"""

import sys
import os
import time
import webbrowser
import subprocess
from pathlib import Path

def main():
    """Run the Wave Theory Streamlit application."""
    print("🌊 Starting Wave Theory Chatbot...")

    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "src" / "app" / "streamlit_app.py"

    if not app_path.exists():
        print(f"❌ Streamlit app not found at: {app_path}")
        return 1

    print(f"📱 Running Streamlit app: {app_path}")
    print("🌐 The app will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)

    proc = None
    try:
        # Detect JAX GPU availability and prefer GPU if present
        gpu_available = False
        try:
            import jax  # Probe backend in this process
            gpu_available = any(d.platform == 'gpu' for d in jax.devices())
            platform = jax.default_backend()
            print(f"🔧 JAX backend detected: {platform}. GPU available: {gpu_available}")
        except Exception:
            print("ℹ️ Could not determine JAX backend; proceeding without forcing platform.")

        env = os.environ.copy()
        # Prefer GPU in the launched app if available and not explicitly overridden
        if gpu_available and not env.get("JAX_PLATFORM_NAME"):
            env["JAX_PLATFORM_NAME"] = "gpu"
            print("⚙️ Preferring GPU for JAX (JAX_PLATFORM_NAME=gpu) in the app process.")

        # Launch Streamlit app
        proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ], env=env)

        # Give Streamlit a moment to start, then open browser
        time.sleep(2)
        try:
            webbrowser.open("http://localhost:8501")
        except Exception:
            pass

        # Wait for Streamlit to exit
        proc.wait()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
