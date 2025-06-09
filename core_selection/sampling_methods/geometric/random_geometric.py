"""
Random sampling in geometric/physics parameter space.
Completely random selection of configurations for baseline comparison.
"""

import numpy as np
from typing import Dict, List, Tuple
from ..base import BaseSampler


class RandomGeometricSampler(BaseSampler):
    """Random sampling in physics parameter space for baseline comparison."""

    def __init__(self, use_6x6_restriction=False, selected_parameters=None):
        self.method_name = "random_geometric"
        super().__init__(use_6x6_restriction=use_6x6_restriction,
                        selected_parameters=selected_parameters)

    def sample(self, n_samples: int, n_runs: int = 10, base_seed: int = 42) -> Dict:
        """
        Perform random geometric sampling multiple times and return the most diverse set.

        Args:
            n_samples: Number of samples to generate
            n_runs: Number of runs to find most diverse set
            base_seed: Base random seed

        Returns:
            Dictionary with sampling results
        """
        print(f"\nRunning Random Geometric {n_runs} times to find most diverse set...")
        print("Completely random selection of configurations")
        print("Provides baseline comparison for geometry-informed methods")

        best_indices = None
        best_diversity = -1
        best_run = 0

        # Store all runs for comparison
        all_runs = []

        for run in range(n_runs):
            # Set seed for reproducibility
            if base_seed is not None:
                np.random.seed(base_seed + run)

            indices = self._single_random_run(n_samples)

            # Calculate diversity in parameter space
            diversity = self.calculate_diversity_score_generic(indices, 'euclidean')

            print(f"  Run {run+1}/{n_runs}: Diversity = {diversity:.4f}")

            # Store run results
            all_runs.append({
                'run': run + 1,
                'indices': indices,
                'diversity': diversity
            })

        # Sort runs by diversity (descending)
        all_runs.sort(key=lambda x: x['diversity'], reverse=True)

        # Select the best run
        best_result = all_runs[0]
        best_indices = best_result['indices']
        best_diversity = best_result['diversity']
        best_run = best_result['run']

        print(f"Best run: #{best_run} with diversity = {best_diversity:.4f}")

        return {
            'selected_indices': best_indices,
            'diversity_score': best_diversity,
            'best_run': best_run,
            'total_runs': n_runs,
            'deterministic': False,
            'method_type': 'random_baseline'
        }

    def _single_random_run(self, n_samples: int) -> List[int]:
        """
        Perform a single random sampling run.
        Simply selects n_samples configurations randomly without replacement.
        """
        n_configs = len(self.physics_parameters)

        # Ensure we don't try to sample more than available
        n_samples = min(n_samples, n_configs)

        # Random selection without replacement
        selected_indices = list(np.random.choice(n_configs, size=n_samples, replace=False))

        return selected_indices
