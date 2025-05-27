"""
Integrated Reactor Model

A comprehensive nuclear reactor simulation package that combines:
- OpenMC neutron transport calculations
- Thermal hydraulics analysis
- Fuel depletion simulations
- Parametric studies
- Comprehensive visualization and plotting

Main entry points:
- main.py: Single simulation execution
- eigenvalue.parametric_study.run_parametric_study(): Parametric studies
"""

__version__ = "1.0.0"
__author__ = "Nuclear Engineering Team"

# Import main functionality
from .inputs import inputs
from .main import main
from .eigenvalue import run_eigenvalue, run_parametric_study

__all__ = ['inputs', 'main', 'run_eigenvalue', 'run_parametric_study']
