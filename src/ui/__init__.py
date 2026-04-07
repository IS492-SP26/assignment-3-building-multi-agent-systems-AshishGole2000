"""
User Interface Module
Contains CLI and web UI implementations.

Usage:
    # Run CLI:
    python -m src.ui.cli
    # or
    python main.py --mode cli
    
    # Run Streamlit web UI:
    streamlit run src/ui/streamlit_app.py
    # or
    python main.py --mode web
"""

from .cli import CLI

__all__ = ["CLI"]
