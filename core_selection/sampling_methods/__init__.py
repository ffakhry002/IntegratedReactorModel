"""
Sampling methods module with organized structure.
"""

# Import from lattice subfolder (configuration space sampling)
from .lattice import (
    LHSLatticeSampler,
    SobolLatticeSampler,
    HaltonLatticeSampler,
    RandomLatticeSampler
)

# Import from geometric subfolder (physics space sampling)
from .geometric import (
    LHSSampler,
    SobolSampler,
    HaltonSampler,
    RandomGeometricSampler
)

__all__ = [
    # Lattice-based methods (configuration space)
    'LHSLatticeSampler',
    'SobolLatticeSampler',
    'HaltonLatticeSampler',
    'RandomLatticeSampler',

    # Geometric/physics-based methods
    'LHSSampler',
    'SobolSampler',
    'HaltonSampler',
    'RandomGeometricSampler'
]
