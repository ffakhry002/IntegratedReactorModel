"""
Greedy max-min diversity algorithm implementation with optimization.
"""
import numpy as np
from typing import List, Optional, Dict, Callable  # <- ADD Callable here
from .base_algorithm import BaseAlgorithm
from .base_algorithm import BaseAlgorithm

class GreedyMaxMin(BaseAlgorithm):
    """Greedy max-min diversity algorithm with vectorized optimizations."""

    def __init__(self):
        super().__init__("greedy")

    def select_samples(self, distance_calculator, n_samples: int,
                    n_items: int, seed: Optional[int] = None,
                    progress_callback: Optional[Callable[[str], None]] = None) -> List[int]:
        """Greedy max-min selection using provided distance calculator with optimization."""
        if seed is not None:
            np.random.seed(seed)

        selected = []
        remaining = set(range(n_items))

        # Start with random configuration
        first_idx = np.random.choice(list(remaining))
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Track minimum distances to selected set for each remaining item
        min_distances = {}

        # Initialize minimum distances using vectorized computation if possible
        if hasattr(distance_calculator, 'feature_matrix') and distance_calculator.feature_matrix is not None:
            # Use vectorized computation
            remaining_list = list(remaining)
            distances = self._vectorized_distances_to_point(
                distance_calculator, remaining_list, first_idx
            )
            for idx, dist in zip(remaining_list, distances):
                min_distances[idx] = dist
        else:
            # Fallback to standard computation
            for idx in remaining:
                min_distances[idx] = distance_calculator.get_distance(idx, first_idx)

        # Progress tracking
        # For small n_samples, report every sample. For large, report every 0.1%
        if n_samples <= 100:
            progress_interval = 1  # Report every sample for small sets
        else:
            progress_interval = max(1, n_samples // 1000)  # Every 0.1% for large sets

        # Greedy selection with optimized distance tracking
        while len(selected) < n_samples and remaining:
            # Progress update
            if len(selected) % progress_interval == 0:
                percent = int((len(selected) / n_samples) * 100)
                print(f"    Progress: {percent}% ({len(selected)}/{n_samples} samples)")

                # ADDED: Call progress callback if provided
                if progress_callback:
                    progress_callback(f"{percent}%")

            # Find candidate with maximum minimum distance
            best_idx = max(remaining, key=lambda x: min_distances[x])
            best_min_dist = min_distances[best_idx]

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

                # Update minimum distances efficiently
                if hasattr(distance_calculator, 'feature_matrix') and distance_calculator.feature_matrix is not None:
                    # Vectorized update
                    remaining_list = list(remaining)
                    if remaining_list:  # Check if any remaining
                        distances_to_new = self._vectorized_distances_to_point(
                            distance_calculator, remaining_list, best_idx
                        )
                        for idx, dist in zip(remaining_list, distances_to_new):
                            min_distances[idx] = min(min_distances[idx], dist)
                else:
                    # Standard update
                    for idx in remaining:
                        dist_to_new = distance_calculator.get_distance(idx, best_idx)
                        min_distances[idx] = min(min_distances[idx], dist_to_new)

                # Remove the selected item from min_distances to save memory
                min_distances.pop(best_idx, None)

        # ADDED: Final progress callback
        if progress_callback:
            progress_callback("100%")

        return selected

    def _vectorized_distances_to_point(self, distance_calculator, indices: List[int],
                                     target_idx: int) -> np.ndarray:
        """
        Compute distances from multiple points to a single target efficiently.

        Args:
            distance_calculator: Distance calculator object
            indices: List of source indices
            target_idx: Target index

        Returns:
            Array of distances
        """
        if not hasattr(distance_calculator, 'feature_matrix') or distance_calculator.feature_matrix is None:
            # Fallback to individual calculations
            return np.array([distance_calculator.get_distance(idx, target_idx) for idx in indices])

        feature_matrix = distance_calculator.feature_matrix
        target_features = feature_matrix[target_idx]
        source_features = feature_matrix[indices]

        # Check if distance calculator has sklearn_metric attribute
        if hasattr(distance_calculator, 'sklearn_metric'):
            metric = distance_calculator.sklearn_metric
        elif hasattr(distance_calculator, 'metric'):
            metric = distance_calculator.metric
        else:
            # Fallback to individual calculations
            return np.array([distance_calculator.get_distance(idx, target_idx) for idx in indices])

        # Use appropriate metric
        if metric == 'euclidean':
            # Efficient Euclidean distance
            distances = np.sqrt(np.sum((source_features - target_features)**2, axis=1))
        elif metric == 'manhattan':
            # Efficient Manhattan distance
            distances = np.sum(np.abs(source_features - target_features), axis=1)
        elif metric == 'cosine':
            # Efficient cosine distance
            # Normalize vectors
            source_norms = np.linalg.norm(source_features, axis=1)
            target_norm = np.linalg.norm(target_features)

            # Compute cosine similarity
            dot_products = np.dot(source_features, target_features)
            cosine_similarities = dot_products / (source_norms * target_norm + 1e-8)

            # Convert to distance
            distances = 1 - cosine_similarities
        else:
            # Fallback for other metrics
            distances = np.array([distance_calculator.get_distance(idx, target_idx) for idx in indices])

        return distances
