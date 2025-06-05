"""
Constants and configuration for sampler execution.
"""

from sampling_methods import (
    # Lattice-based methods (configuration space)
    LHSLatticeSampler,
    SobolLatticeSampler,
    HaltonLatticeSampler,
    RandomLatticeSampler,

    # Geometric/physics-based methods
    LHSSampler,
    SobolSampler,
    HaltonSampler,
    RandomGeometricSampler
)
from sampling_methods.geometric import lhs, sobol, halton, random_geometric
from sampling_methods.lattice import (
    lhs_lattice, sobol_lattice, halton_lattice, random_lattice
)
from sampling_methods.unified import GeometricUnifiedSampler, LatticeUnifiedSampler


# Global sampler map for all execution modes
def get_sampler_map(use_6x6_restriction=False):
    """Get sampler map with optional 6x6 restriction."""
    return {
        # Sequence-based methods (LHS, Sobol, Halton)
        'lhs': lambda: lhs.LHSSampler(use_6x6_restriction=use_6x6_restriction),
        'sobol': lambda: sobol.SobolSampler(use_6x6_restriction=use_6x6_restriction),
        'halton': lambda: halton.HaltonSampler(use_6x6_restriction=use_6x6_restriction),
        'random_geometric': lambda: random_geometric.RandomGeometricSampler(use_6x6_restriction=use_6x6_restriction),

        'lhs_lattice': lambda: lhs_lattice.LHSLatticeSampler(use_6x6_restriction=use_6x6_restriction),
        'sobol_lattice': lambda: sobol_lattice.SobolLatticeSampler(use_6x6_restriction=use_6x6_restriction),
        'halton_lattice': lambda: halton_lattice.HaltonLatticeSampler(use_6x6_restriction=use_6x6_restriction),
        'random_lattice': lambda: random_lattice.RandomLatticeSampler(use_6x6_restriction=use_6x6_restriction),

        # Greedy max-min algorithms with different distances
        'euclidean_geometric': lambda: GeometricUnifiedSampler('greedy', 'euclidean', use_6x6_restriction=use_6x6_restriction),
        'manhattan_geometric': lambda: GeometricUnifiedSampler('greedy', 'manhattan', use_6x6_restriction=use_6x6_restriction),
        'jaccard_geometric': lambda: GeometricUnifiedSampler('greedy', 'jaccard', use_6x6_restriction=use_6x6_restriction),

        'euclidean_lattice': lambda: LatticeUnifiedSampler('greedy', 'euclidean', use_6x6_restriction=use_6x6_restriction),
        'manhattan_lattice': lambda: LatticeUnifiedSampler('greedy', 'manhattan', use_6x6_restriction=use_6x6_restriction),
        'jaccard_lattice': lambda: LatticeUnifiedSampler('greedy', 'jaccard', use_6x6_restriction=use_6x6_restriction),

        # NEW: Euclidean lattice with geometric diversity evaluation
        'euclidean_lattice_geometric_diversity': lambda: LatticeUnifiedSampler('greedy', 'euclidean', use_geometric_diversity=True, use_6x6_restriction=use_6x6_restriction),

        # K-means algorithms (method IDs kept as 'kmedoids' for backwards compatibility)
        # These use k-means clustering to find centers, then select nearest actual points
        # IMPORTANT: Despite the method names, these are K-MEANS algorithms, NOT k-medoids!
        'euclidean_geometric_kmedoids': lambda: GeometricUnifiedSampler('kmedoids', 'euclidean', use_6x6_restriction=use_6x6_restriction),
        # 'manhattan_geometric_kmedoids': lambda: GeometricUnifiedSampler('kmedoids', 'manhattan'),
        # 'jaccard_geometric_kmedoids': lambda: GeometricUnifiedSampler('kmedoids', 'jaccard'),

        'euclidean_lattice_kmedoids': lambda: LatticeUnifiedSampler('kmedoids', 'euclidean', use_6x6_restriction=use_6x6_restriction),
        # 'manhattan_lattice_kmedoids': lambda: LatticeUnifiedSampler('kmedoids', 'manhattan'),
        # 'jaccard_lattice_kmedoids': lambda: LatticeUnifiedSampler('kmedoids', 'jaccard'),
    }

# Keep default SAMPLER_MAP for backward compatibility
SAMPLER_MAP = get_sampler_map()


# Check if Rich is available for progress display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
