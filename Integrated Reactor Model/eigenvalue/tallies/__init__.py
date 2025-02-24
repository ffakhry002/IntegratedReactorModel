"""
This module contains functions for creating and managing OpenMC tallies.
"""

from .irradiation_tallies import create_irradiation_tallies, create_irradiation_axial_tallies
from .core_tallies import create_nutotal_tallies, create_coreflux_tallys

__all__ = [
    'create_irradiation_tallies',
    'create_irradiation_axial_tallies',
    'create_nutotal_tallies',
    'create_coreflux_tallys'
]
