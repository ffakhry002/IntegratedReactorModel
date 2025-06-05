#!/usr/bin/env python3
"""
Main Launcher for Parametric GUI
Entry point for the Reactor Model Parametric Study Generator
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the standalone GUI
from parametric_gui_standalone import main

if __name__ == "__main__":
    print("Starting Reactor Model - Parametric Study Generator...")
    print("=" * 60)
    print("Features:")
    print("• Simple Parameter Studies")
    print("• Multi-Parameter Loops")
    print("• Visual Core Lattice Designer")
    print("• Export to run_dictionaries.py")
    print("=" * 60)
    print()

    main()
