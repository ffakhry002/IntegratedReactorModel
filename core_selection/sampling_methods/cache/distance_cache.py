"""
Efficient distance caching with LRU eviction for sampling methods.
"""
from typing import Dict, Tuple, Callable
from collections import defaultdict


class EfficientDistanceCache:
    """Memory-efficient distance cache with LRU eviction and smart precomputation."""

    def __init__(self, max_cache_size=100000):
        self.max_cache_size = max_cache_size
        self.cache_stats = defaultdict(int)

        # Separate caches for different distance types
        self.euclidean_cache = {}
        self.manhattan_cache = {}
        self.jaccard_cache = {}
        self.lattice_euclidean_cache = {}
        self.lattice_manhattan_cache = {}

        # Priority tracking for LRU eviction
        self.access_count = defaultdict(int)
        self.access_order = []

    def _get_cache_key(self, idx1: int, idx2: int) -> Tuple[int, int]:
        """Generate symmetric cache key."""
        return tuple(sorted([idx1, idx2]))

    def _evict_lru_if_needed(self, cache_dict: Dict):
        """Evict least recently used items if cache is too large."""
        if len(cache_dict) > self.max_cache_size:
            # Find LRU items
            items_by_access = [(self.access_count[key], key) for key in cache_dict.keys()]
            items_by_access.sort()  # Sort by access count (ascending)

            # Remove 20% of least accessed items
            n_to_remove = max(1, len(items_by_access) // 5)
            for _, key in items_by_access[:n_to_remove]:
                cache_dict.pop(key, None)
                self.access_count.pop(key, None)

    def get_distance(self, distance_type: str, idx1: int, idx2: int,
                    compute_func: Callable) -> float:
        """Get cached distance or compute and cache."""
        key = self._get_cache_key(idx1, idx2)

        # Select appropriate cache
        cache_dict = getattr(self, f"{distance_type}_cache")

        if key in cache_dict:
            self.access_count[key] += 1
            self.cache_stats[f"{distance_type}_hits"] += 1
            return cache_dict[key]

        # Compute distance
        distance = compute_func(idx1, idx2)

        # Cache the result
        cache_dict[key] = distance
        self.access_count[key] = 1
        self.cache_stats[f"{distance_type}_computes"] += 1

        # Evict if necessary
        self._evict_lru_if_needed(cache_dict)

        return distance

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_size = (len(self.euclidean_cache) +
                     len(self.manhattan_cache) +
                     len(self.jaccard_cache) +
                     len(self.lattice_euclidean_cache) +
                     len(self.lattice_manhattan_cache))

        return {
            'total_cached_distances': total_size,
            'cache_hit_rates': {
                distance_type: (
                    self.cache_stats[f"{distance_type}_hits"] /
                    max(1, self.cache_stats[f"{distance_type}_hits"] +
                        self.cache_stats[f"{distance_type}_computes"])
                ) for distance_type in ['euclidean', 'manhattan', 'jaccard', 'lattice_euclidean', 'lattice_manhattan']
            },
            'individual_cache_sizes': {
                'euclidean': len(self.euclidean_cache),
                'manhattan': len(self.manhattan_cache),
                'jaccard': len(self.jaccard_cache),
                'lattice_euclidean': len(self.lattice_euclidean_cache),
                'lattice_manhattan': len(self.lattice_manhattan_cache)
            }
        }

    def clear(self):
        """Clear all caches."""
        self.euclidean_cache.clear()
        self.manhattan_cache.clear()
        self.jaccard_cache.clear()
        self.lattice_euclidean_cache.clear()
        self.lattice_manhattan_cache.clear()
        self.access_count.clear()
        self.cache_stats.clear()
