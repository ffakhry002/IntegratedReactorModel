"""
Parametric2_GUI Package
A GUI for creating and managing parametric studies for the Integrated Reactor Model

This package provides tools for creating and configuring parametric studies
for the Integrated Reactor Model.

Main Components:
- parametric_app.py: Core parametric study configuration interface
- components/: Tab components for different study types
- designers/: Visual designers for complex parameters
- models/: Data models for parameters and configurations
- utils/: Utility functions

Usage:
- Run parametric_gui_standalone.py for standalone operation
- Import ParametricApp for integration into other applications
"""

from .parametric_app import ParametricApp
from .parametric_gui_standalone import main, ParametricGUIStandalone

__version__ = "2.0.0"
__author__ = "Reactor Model Team"

__all__ = ['ParametricApp', 'main', 'ParametricGUIStandalone']
