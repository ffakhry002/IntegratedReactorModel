"""
Lattice distance calculations with symmetry awareness.
"""
from .base_distance import BaseDistance

class LatticeEuclidean(BaseDistance):
    """Symmetry-aware Euclidean distance for lattice positions."""

    def __init__(self, base_sampler):
        super().__init__("euclidean")
        self.sampler = base_sampler
        self.supports_feature_space = False  # Must use distance matrix

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Calculate symmetry-aware Euclidean distance WITH CACHING."""
        # Use the cached version instead!
        return self.sampler.get_cached_lattice_euclidean_distance(idx1, idx2)


class LatticeManhattan(BaseDistance):
    """Symmetry-aware Manhattan distance for lattice positions."""

    def __init__(self, base_sampler):
        super().__init__("manhattan")
        self.sampler = base_sampler
        self.supports_feature_space = False

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Calculate symmetry-aware Manhattan distance WITH CACHING."""
        # Use the cached version instead!
        return self.sampler.get_cached_lattice_manhattan_distance(idx1, idx2)


class LatticeJaccard(BaseDistance):
    """Symmetry-aware Jaccard distance for lattice positions."""

    def __init__(self, base_sampler):
        super().__init__("jaccard")
        self.sampler = base_sampler
        self.supports_feature_space = False

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Calculate symmetry-aware Jaccard distance."""
        # Note: This already uses caching internally via calculate_symmetry_aware_jaccard_distance
        return self.sampler.calculate_symmetry_aware_jaccard_distance(idx1, idx2)
