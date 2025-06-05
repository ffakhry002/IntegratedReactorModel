"""
Base class for Quasi-Monte Carlo (QMC) samplers with common functionality.
Place this file in the same directory as the QMC samplers.
"""

import numpy as np
from typing import Dict, List, Tuple
from ..base import BaseSampler


class BaseQMCSampler(BaseSampler):
    """Base class for QMC samplers with common functionality."""

    def __init__(self, use_6x6_restriction=False):
        super().__init__(use_6x6_restriction=use_6x6_restriction)

    def find_closest_configuration(self, target_params: np.ndarray,
                                 used_indices: set = None) -> Tuple[int, float]:
        """Find the configuration whose parameters are closest to the target."""
        if used_indices is None:
            used_indices = set()

        # Calculate Euclidean distances to all configurations
        distances = np.linalg.norm(self.feature_matrix_normalized - target_params, axis=1)

        # Mask out already used configurations
        for idx in used_indices:
            distances[idx] = np.inf

        # Find the closest configuration
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]

        return closest_idx, min_distance

    def _get_parameter_ranges(self):
        """Get the parameter ranges for scaling."""
        n_features = self.feature_matrix.shape[1]
        param_ranges = []
        for j in range(n_features):
            param_ranges.append((
                self.feature_matrix[:, j].min(),
                self.feature_matrix[:, j].max()
            ))
        return param_ranges

    def _scale_samples_to_ranges(self, samples_unit: np.ndarray, param_ranges: List[Tuple[float, float]]) -> np.ndarray:
        """Scale unit cube samples to actual parameter ranges."""
        n_features = len(param_ranges)
        scaled_samples = np.zeros_like(samples_unit)

        for j in range(n_features):
            min_val, max_val = param_ranges[j]
            scaled_samples[:, j] = samples_unit[:, j] * (max_val - min_val) + min_val

        return scaled_samples

    def _find_closest_configurations(self, normalized_samples: np.ndarray, n_samples: int) -> Tuple[List[int], List[float]]:
        """Find closest discrete configurations to continuous samples."""
        selected_indices = []
        distances = []
        used_indices = set()

        for i in range(n_samples):
            target = normalized_samples[i]
            closest_idx, min_dist = self.find_closest_configuration(target, used_indices)
            selected_indices.append(int(closest_idx))
            distances.append(float(min_dist))
            used_indices.add(closest_idx)

        return selected_indices, distances

    def _run_multiple_times(self, n_samples: int, n_runs: int, base_seed: int,
                           sampler_name: str, single_run_func) -> Dict:
        """Common logic for running QMC samplers multiple times."""
        print(f"\nRunning {sampler_name} {n_runs} times to find most diverse set...")

        best_indices = None
        best_diversity = -1
        best_avg_distance = float('inf')
        best_run = 0

        # Store all runs for comparison
        all_runs = []

        for run in range(n_runs):
            # Use different seed for each run
            seed = base_seed + run if base_seed is not None else None

            indices, distances = single_run_func(n_samples, seed)

            # Calculate diversity in parameter space
            diversity = self.calculate_diversity_score_generic(indices, 'euclidean')
            avg_distance = np.mean(distances)

            print(f"  Run {run+1}/{n_runs}: Avg Distance = {avg_distance:.4f}, Diversity = {diversity:.4f}")

            # Store run results
            all_runs.append({
                'run': run + 1,
                'indices': indices,
                'distances': distances,
                'avg_distance': avg_distance,
                'diversity': diversity
            })

        # Sort runs by diversity (descending) then by avg_distance (ascending)
        # This prioritizes diversity while using avg_distance as tiebreaker
        all_runs.sort(key=lambda x: (x['diversity'], -x['avg_distance']), reverse=True)

        # Select the best run
        best_result = all_runs[0]
        best_indices = best_result['indices']
        best_diversity = best_result['diversity']
        best_avg_distance = best_result['avg_distance']
        best_run = best_result['run']

        print(f"Best run: #{best_run} with diversity = {best_diversity:.4f} and avg distance = {best_avg_distance:.4f}")

        return {
            'selected_indices': best_indices,
            'matching_distances': best_result['distances'],
            'diversity_score': best_diversity,
            'avg_matching_distance': float(best_avg_distance),
            'best_run': best_run,
            'total_runs': n_runs
        }
