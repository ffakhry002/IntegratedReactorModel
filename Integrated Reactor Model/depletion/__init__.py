"""
Depletion package for running and analyzing depletion calculations.

This package provides functionality for performing depletion calculations
on full reactor core models, tracking k-effective and nuclide concentrations
over time.
"""

from .run_depletion import run_depletion
from .depletion_operator import (
    create_operator,
    setup_depletion,
)

__all__ = [
    'create_operator',
    'setup_depletion',
    'run_depletion'
]
