#!/usr/bin/env python3
"""
Wave Theory Chatbot - Application Runner
Simple script to run the Streamlit application
"""

import sys
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
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
