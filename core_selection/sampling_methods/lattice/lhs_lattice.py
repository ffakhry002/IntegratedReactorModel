"""
Latin Hypercube Sampling directly in 4D lattice configuration space.
Uses symmetry-reduced set with 8-way rotation matching.
Includes proper forbidden position handling.
"""

import numpy as np
from scipy.stats import qmc
from typing import Dict, List, Tuple
from .base_lattice import BaseLatticeSampler


class LHSLatticeSampler(BaseLatticeSampler):
    """Latin Hypercube Sampling in lattice space with symmetry awareness."""

    def __init__(self, use_6x6_restriction=False):
        super().__init__(use_6x6_restriction=use_6x6_restriction)
        self.method_name = "lhs_lattice"

    def sample(self, n_samples: int, n_runs: int = 10, base_seed: int = 42) -> Dict:
        """
        Perform LHS lattice sampling with symmetry-aware matching.

        Args:
            n_samples: Number of samples to generate
            n_runs: Number of runs to find most diverse set
            base_seed: Base random seed

        Returns:
            Dictionary with sampling results
        """
        print(f"\nRunning LHS Lattice {n_runs} times...")
        print("Using symmetry-reduced configuration set with 8-way rotation matching")
        print("With proper forbidden position handling")

        best_indices = None
        best_diversity = -1
        best_avg_distance = -1
        best_run = 0

        # Store all runs for comparison
        all_runs = []

        for run in range(n_runs):
            # Set seed for reproducibility
            if base_seed is not None:
                seed = base_seed + run
            else:
                seed = None

            indices, avg_distance = self._single_lhs_lattice_run(n_samples, seed)

            # Calculate diversity in physics parameter space
            diversity = self.calculate_diversity_score_lattice_generic(indices, 'euclidean')


            print(f"  Run {run+1}/{n_runs}: Avg Distance = {avg_distance:.4f}, Diversity = {diversity:.4f}")

            # Store run results
            all_runs.append({
                'run': run + 1,
                'indices': indices,
                'avg_distance': avg_distance,
                'diversity': diversity
            })

        # Sort runs by diversity (descending) then by avg_distance (descending)
        all_runs.sort(key=lambda x: (x['diversity'], x['avg_distance']), reverse=True)

        # Select the best run
        best_result = all_runs[0]
        best_indices = best_result['indices']
        best_diversity = best_result['diversity']
        best_avg_distance = best_result['avg_distance']
        best_run = best_result['run']

        print(f"Best run: #{best_run} with diversity = {best_diversity:.4f} and avg distance = {best_avg_distance:.4f}")

        return {
            'selected_indices': best_indices,
            'diversity_score': best_diversity,
            'avg_lattice_distance': best_avg_distance,
            'best_run': best_run,
            'total_runs': n_runs,
            'symmetry_aware': True,
            'forbidden_aware': True
        }

    def _single_lhs_lattice_run(self, n_samples: int, seed: int = None) -> Tuple[List[int], float]:
        """Perform sampling using true 8D continuous space with forbidden position handling."""
        # Generate samples in 8D space (4 positions × 2 coordinates each)
        sampler = qmc.LatinHypercube(d=8, seed=seed)
        samples_8d = sampler.random(n=n_samples)

        # Convert each 8D sample to 4 lattice positions
        selected_indices = []
        lattice_distances = []
        used_configs = set()

        for i in range(n_samples):
            # Convert 8D sample to 4 valid positions
            positions = []
            for j in range(4):
                x = samples_8d[i, 2*j]
                y = samples_8d[i, 2*j + 1]
                # Map to nearest valid fuel position (avoiding forbidden positions)
                valid_pos = self._continuous_to_valid_position(x, y)
                positions.append(valid_pos)

            # Find best matching configuration with symmetry awareness
            best_config_idx = self._find_best_symmetry_match(positions, used_configs)

            if best_config_idx is None:
                # Fallback: find any unused configuration
                for idx in range(len(self.irradiation_sets)):
                    if idx not in used_configs:
                        best_config_idx = idx
                        break

            if best_config_idx is not None:
                selected_indices.append(best_config_idx)
                used_configs.add(best_config_idx)

                # Calculate lattice distance
                if i > 0:
                    prev_positions = set(self.irradiation_sets[selected_indices[i-1]])
                    curr_positions = set(self.irradiation_sets[best_config_idx])
                    lattice_dist = len(prev_positions.symmetric_difference(curr_positions))
                    lattice_distances.append(lattice_dist)

        avg_distance = np.mean(lattice_distances) if lattice_distances else 0.0
        return selected_indices, avg_distance
