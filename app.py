#!/usr/bin/env python3
"""
Wave Theory Chatbot - Docker Entry Point
Main application entry point for containerized deployment
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the Streamlit app
if __name__ == "__main__":
    from src.app.streamlit_app import main
    main()
