"""
Eigenvalue calculations and parametric studies module.

This module contains OpenMC transport calculations and parametric study functionality.
"""

from .run import run_eigenvalue
from .parametric_study import run_parametric_study
from .outputs import process_results

__all__ = ['run_eigenvalue', 'run_parametric_study', 'process_results']
