"""
Unified lattice sampler with configurable algorithm and distance.
UPDATED: Now calculates diversity using the same metric as selection in lattice space
"""
from typing import Dict, Tuple
import numpy as np
from ..base import BaseSampler
from ..algorithms import GreedyMaxMin, KMeansNearestAlgorithm
from ..distances.lattice_distances import (
    LatticeEuclidean, LatticeManhattan, LatticeJaccard
)

class LatticeUnifiedSampler(BaseSampler):
    """Unified lattice sampler with configurable algorithm and distance."""

    def __init__(self, algorithm='greedy', distance='euclidean', use_geometric_diversity=False, use_6x6_restriction=False):
        # Set method name BEFORE calling super().__init__() so base class loads correct dataset
        if use_geometric_diversity:
            self.method_name = f"{distance}_lattice_{algorithm}_geometric_diversity"
        else:
            self.method_name = f"{distance}_lattice_{algorithm}"

        super().__init__(use_6x6_restriction=use_6x6_restriction)

        # Store the diversity calculation mode
        self.use_geometric_diversity = use_geometric_diversity

        # Initialize algorithm
        if algorithm == 'greedy':
            self.algorithm = GreedyMaxMin()
        elif algorithm == 'kmedoids':
            self.algorithm = KMeansNearestAlgorithm(n_init=1)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Initialize distance calculator
        if distance == 'euclidean':
            self.distance_calculator = LatticeEuclidean(self)
        elif distance == 'manhattan':
            self.distance_calculator = LatticeManhattan(self)
        elif distance == 'jaccard':
            self.distance_calculator = LatticeJaccard(self)
        else:
            raise ValueError(f"Unknown distance: {distance}")

    def _discretize_features(self, features):
        """
        Use unified discretization method from base class.
        This ensures consistency with other sampling methods.
        """
        return self._discretize_features_unified(features)

    def _run_single_sample(self, n_samples: int, seed: int) -> Tuple:
        """Run a single sampling iteration (for parallel execution)."""
        # Check if we have a progress callback to pass along
        progress_callback = getattr(self, '_progress_callback', None)

        # If we have a callback and the algorithm supports it, set it on the algorithm
        if progress_callback and hasattr(self.algorithm, 'select_samples'):
            if 'progress_callback' in self.algorithm.select_samples.__code__.co_varnames:
                # Set it on the algorithm instance for methods that check self._progress_callback
                self.algorithm._progress_callback = progress_callback

        # Run algorithm with progress callback if supported
        kwargs = {
            'distance_calculator': self.distance_calculator,
            'n_samples': n_samples,
            'n_items': len(self.irradiation_sets),
            'seed': seed
        }

        # Add progress callback if the algorithm supports it
        if progress_callback and 'progress_callback' in self.algorithm.select_samples.__code__.co_varnames:
            kwargs['progress_callback'] = progress_callback

        indices = self.algorithm.select_samples(**kwargs)

        # Get quality metric if available
        if self.algorithm.supports_quality_metric():
            quality_value, _ = self.algorithm.get_quality_metric()
        else:
            # UPDATED: Use geometric diversity if specified, otherwise use lattice diversity
            if self.use_geometric_diversity:
                quality_value = self.calculate_diversity_score_generic(indices, 'euclidean')
            else:
                quality_value = self.calculate_diversity_score_lattice_generic(indices, self.distance_calculator.name)


        return indices, quality_value

    def sample(self, n_samples: int, n_runs: int = 10, base_seed: int = 42) -> Dict:
        """Run sampling with configured algorithm and distance."""
        print(f"\nRunning {self.method_name}")
        print(f"Algorithm: {self.algorithm.name}")
        print(f"Distance: {self.distance_calculator.name}")
        if self.use_geometric_diversity:
            print(f"Diversity calculation: Geometric space (5D physics parameters)")
        else:
            print(f"Diversity calculation: Lattice space")
        print(f"Total configurations: {len(self.irradiation_sets)}")

        # Handle single run case (for parallel execution)
        if n_runs == 1:
            seed = base_seed if base_seed else None
            indices, quality = self._run_single_sample(n_samples, seed)
            # UPDATED: Use appropriate diversity calculation
            if self.use_geometric_diversity:
                diversity = self.calculate_diversity_score_generic(indices, 'euclidean')
            else:
                diversity = self.calculate_diversity_score_lattice_generic(indices, self.distance_calculator.name)

            result = {
                'selected_indices': indices,
                'diversity_score': diversity,
                'best_run': 1,
                'total_runs': 1,
                'all_diversities': [diversity],
                'algorithm': self.algorithm.name,
                'distance': self.distance_calculator.name,
                'selection_metric': 'inertia' if self.algorithm.name == 'kmeans_nearest' else 'diversity'
            }

            # Add inertia for k-means
            if self.algorithm.name == 'kmeans_nearest':
                result['inertia'] = quality

            return result

        # Multiple runs - full logic
        best_indices = None
        best_score = -1 if self.algorithm.name == 'greedy' else float('inf')
        best_diversity = -1
        best_run = 0
        all_diversities = []

        # For k-means, track inertia
        run_stats = []
        all_inertia_scores = []

        for run in range(n_runs):
            seed = base_seed + run if base_seed else None

            # Run algorithm
            kwargs = {
                'distance_calculator': self.distance_calculator,
                'n_samples': n_samples,
                'n_items': len(self.irradiation_sets),
                'seed': seed
            }

            # Add progress callback if available
            progress_callback = getattr(self, '_progress_callback', None)
            if progress_callback and 'progress_callback' in self.algorithm.select_samples.__code__.co_varnames:
                kwargs['progress_callback'] = progress_callback

            indices = self.algorithm.select_samples(**kwargs)

            # UPDATED: Calculate diversity using appropriate space metric
            if self.use_geometric_diversity:
                diversity = self.calculate_diversity_score_generic(indices, 'euclidean')
            else:
                diversity = self.calculate_diversity_score_lattice_generic(indices, self.distance_calculator.name)
            all_diversities.append(diversity)

            # For k-means, get inertia
            if self.algorithm.supports_quality_metric():
                quality_value, quality_name = self.algorithm.get_quality_metric()
                print(f"  Run {run+1}/{n_runs}: {quality_name} = {quality_value:.2f}, Diversity = {diversity:.4f}")

                run_stats.append({
                    'run': run + 1,
                    'indices': indices,
                    'diversity': diversity,
                    quality_name: quality_value
                })

                # Store inertia score for k-means
                if self.algorithm.name == 'kmeans_nearest':
                    all_inertia_scores.append(quality_value)

                # For k-means, select based on lowest inertia
                if quality_value < best_score:
                    best_score = quality_value
                    best_indices = indices
                    best_diversity = diversity
                    best_run = run + 1
            else:
                # For greedy, select based on highest diversity
                print(f"  Run {run+1}/{n_runs}: Diversity = {diversity:.4f}")

                if diversity > best_score:
                    best_score = diversity
                    best_indices = indices
                    best_diversity = diversity
                    best_run = run + 1

        # Print selection summary for k-means
        if self.algorithm.name == 'kmeans_nearest' and run_stats:
            print(f"\nK-Means Selection Summary:")
            print(f"Selected run {best_run} with lowest inertia = {best_score:.2f}")
            print(f"This run had diversity = {best_diversity:.4f}")

            # Show if different from max diversity
            max_div_run = max(run_stats, key=lambda x: x['diversity'])
            if max_div_run['run'] != best_run:
                print(f"Note: Run {max_div_run['run']} had highest diversity ({max_div_run['diversity']:.4f}) but worse clustering")

        result = {
            'selected_indices': best_indices,
            'diversity_score': best_diversity,
            'best_run': best_run,
            'total_runs': n_runs,
            'all_diversities': all_diversities,
            'algorithm': self.algorithm.name,
            'distance': self.distance_calculator.name,
            'selection_metric': 'inertia' if self.algorithm.name == 'kmeans_nearest' else 'diversity'
        }

        # Add inertia for k-means
        if self.algorithm.name == 'kmeans_nearest':
            result['inertia'] = best_score
            result['all_inertia_scores'] = all_inertia_scores

        return result

    def calculate_diversity_score_lattice(self, selected_indices) -> float:
        return self.calculate_diversity_score_lattice_generic(selected_indices, self.distance_calculator.name)
