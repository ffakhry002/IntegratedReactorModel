"""
Parallel base class for sampling methods optimized for HPC environments.
Uses multiprocessing to parallelize expensive distance calculations.
"""

import numpy as np
import pickle
import json
from typing import List, Tuple, Dict, Set
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from functools import partial
import os

from .symmetry_utils import ALL_SYMMETRIES, are_configurations_symmetric, get_canonical_form
from .base import BaseSampler


class ParallelBaseSampler(BaseSampler):
    """Parallel version of BaseSampler with multiprocessing support."""

    def __init__(self, n_workers=None):
        """Initialize with specified number of workers."""
        self.n_workers = n_workers if n_workers else cpu_count()
        print(f"  Initializing parallel sampler with {self.n_workers} workers")
        super().__init__()

    def parallel_precompute_distances(self, indices: List[int], distance_type: str = 'euclidean'):
        """Precompute distances in parallel for a subset of configurations."""
        print(f"  Parallel precomputing {distance_type} distances for {len(indices)} configurations...")

        # Create pairs to compute
        pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.append((indices[i], indices[j]))

        # Prepare function based on distance type
        if distance_type == 'euclidean':
            compute_func = self._compute_euclidean_distance_pair
        elif distance_type == 'manhattan':
            compute_func = self._compute_manhattan_distance_pair
        elif distance_type == 'jaccard':
            compute_func = self._compute_jaccard_distance_pair
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")

        # Compute in parallel
        with Pool(self.n_workers) as pool:
            results = pool.map(compute_func, pairs)

        # Store results in cache
        cache_dict = {
            'euclidean': self.euclidean_cache,
            'manhattan': self.manhattan_cache,
            'jaccard': self.distance_cache
        }[distance_type]

        for (idx1, idx2), dist in zip(pairs, results):
            key = self._get_cache_key(idx1, idx2)
            cache_dict[key] = dist

        print(f"    ✓ Computed {len(results)} distances in parallel")

    def _compute_euclidean_distance_pair(self, pair):
        """Compute Euclidean distance for a single pair."""
        idx1, idx2 = pair
        dist = np.linalg.norm(
            self.feature_matrix_normalized[idx1] -
            self.feature_matrix_normalized[idx2]
        )
        return dist

    def _compute_manhattan_distance_pair(self, pair):
        """Compute Manhattan distance for a single pair."""
        idx1, idx2 = pair
        dist = np.sum(np.abs(
            self.feature_matrix_normalized[idx1] -
            self.feature_matrix_normalized[idx2]
        ))
        return dist

    def _compute_jaccard_distance_pair(self, pair):
        """Compute Jaccard distance for a single pair."""
        idx1, idx2 = pair
        if self.position_sets is None:
            self._precompute_optimization_data()
        return self.calculate_jaccard_distance(
            self.position_sets[idx1],
            self.position_sets[idx2]
        )

    def parallel_find_best_candidates(self, candidates: List[int],
                                    selected: List[int],
                                    distance_func) -> Tuple[int, float]:
        """Find best candidate in parallel by computing distances to all selected."""
        if len(candidates) < 100:  # Not worth parallelizing for small sets
            # Fall back to serial computation
            best_idx = None
            max_min_distance = -1

            for candidate_idx in candidates:
                min_distance = float('inf')
                for selected_idx in selected:
                    dist = distance_func(candidate_idx, selected_idx)
                    min_distance = min(min_distance, dist)

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx

            return best_idx, max_min_distance

        # Parallel computation for large candidate sets
        compute_func = partial(self._compute_min_distance_to_selected,
                             selected=selected,
                             distance_func=distance_func)

        with Pool(self.n_workers) as pool:
            results = pool.map(compute_func, candidates)

        # Find best candidate
        best_idx = None
        max_min_distance = -1
        for candidate_idx, min_distance in zip(candidates, results):
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_idx = candidate_idx

        return best_idx, max_min_distance

    def _compute_min_distance_to_selected(self, candidate_idx, selected, distance_func):
        """Compute minimum distance from candidate to all selected configs."""
        min_distance = float('inf')
        for selected_idx in selected:
            dist = distance_func(candidate_idx, selected_idx)
            min_distance = min(min_distance, dist)
        return min_distance


class ParallelGreedySampler(ParallelBaseSampler):
    """Base class for parallel greedy max-min samplers."""

    def parallel_greedy_selection(self, n_samples: int, distance_func,
                                distance_type: str = 'euclidean') -> Tuple[List[int], float]:
        """Parallel version of greedy max-min selection."""
        selected_indices = []
        remaining_indices = list(range(len(self.physics_parameters)))

        # Start with a random configuration
        first_idx = int(np.random.choice(remaining_indices))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Track minimum distance
        min_distance = float('inf')

        # Progress tracking
        print(f"  Parallel greedy selection ({distance_type}):")

        # Greedily select configurations
        while len(selected_indices) < n_samples and remaining_indices:
            # Show progress
            if len(selected_indices) % 5 == 0:
                print(f"    Selected {len(selected_indices)}/{n_samples} configurations...")

            # Precompute distances for current batch if not cached
            if len(selected_indices) < 10:  # Early stages, precompute more
                batch_size = min(1000, len(remaining_indices))
                batch = remaining_indices[:batch_size]
                self.parallel_precompute_distances(
                    selected_indices + batch,
                    distance_type
                )

            # Find best candidate in parallel
            best_idx, max_min_distance = self.parallel_find_best_candidates(
                remaining_indices,
                selected_indices,
                distance_func
            )

            if best_idx is not None:
                selected_indices.append(int(best_idx))
                remaining_indices.remove(best_idx)
                min_distance = min(min_distance, max_min_distance)

        print(f"    ✓ Selected {len(selected_indices)} configurations")
        return selected_indices, min_distance
