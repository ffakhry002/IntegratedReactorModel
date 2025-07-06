"""
Unified geometric sampler with configurable algorithm and distance.
"""
from typing import Dict, Tuple
import numpy as np
from ..base import BaseSampler
from ..algorithms import GreedyMaxMin, KMeansNearestAlgorithm
from ..distances.geometric_distances import (
    GeometricEuclidean, GeometricManhattan, GeometricJaccard
)

class GeometricUnifiedSampler(BaseSampler):
    """Unified geometric sampler with configurable algorithm and distance."""

    def __init__(self, algorithm='greedy', distance='euclidean', use_6x6_restriction=False, selected_parameters=None):
        # Set method name BEFORE calling super().__init__() so base class loads correct dataset
        self.method_name = f"{distance}_geometric_{algorithm}"

        super().__init__(use_6x6_restriction=use_6x6_restriction,
                        selected_parameters=selected_parameters)

        # Store config for later use
        self.algorithm_name = algorithm
        self.distance_name = distance

        # Initialize algorithm
        if algorithm == 'greedy':
            self.algorithm = GreedyMaxMin()
        elif algorithm == 'kmedoids':
            # Use k-means with nearest neighbor selection instead of k-medoids
            self.algorithm = KMeansNearestAlgorithm(n_init=1)  # Single run per iteration
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Initialize distance calculator
        if distance == 'euclidean':
            self.distance_calculator = GeometricEuclidean(self)
        elif distance == 'manhattan':
            self.distance_calculator = GeometricManhattan(self)
        elif distance == 'jaccard':
            self.distance_calculator = GeometricJaccard(self)
        else:
            raise ValueError(f"Unknown distance: {distance}")

    def _discretize_features(self, features):
        """Use unified discretization method from base class.

        This ensures consistency with other sampling methods.

        Parameters
        ----------
        features : np.ndarray
            Features to discretize

        Returns
        -------
        np.ndarray
            Discretized features
        """
        return self._discretize_features_unified(features)

    def _run_single_sample(self, n_samples: int, seed: int) -> Tuple:
        """Run a single sampling iteration (for parallel execution).

        Parameters
        ----------
        n_samples : int
            Number of samples to select
        seed : int
            Random seed for reproducibility

        Returns
        -------
        Tuple
            (selected_indices, quality_value) where quality_value is either inertia or diversity
        """
        # Check if we have a progress callback to pass
        progress_callback = getattr(self, '_progress_callback', None)

        # Run algorithm
        indices = self.algorithm.select_samples(
            self.distance_calculator,
            n_samples,
            len(self.physics_parameters),
            seed,
            progress_callback=progress_callback if progress_callback else None
        )

        # Get quality metric if available
        if self.algorithm.supports_quality_metric():
            quality_value, _ = self.algorithm.get_quality_metric()
        else:
            quality_value = self.calculate_diversity_score_generic(indices, self.distance_calculator.name)

        # ADDED: Store cluster assignments if k-means
        if hasattr(self.algorithm, 'cluster_assignments'):
            self.cluster_assignments = self.algorithm.cluster_assignments
            self.n_clusters = self.algorithm.n_clusters

        return indices, quality_value

    def sample(self, n_samples: int, n_runs: int = 10, base_seed: int = 42) -> Dict:
        """Run sampling with configured algorithm and distance.

        Parameters
        ----------
        n_samples : int
            Number of samples to select
        n_runs : int, optional
            Number of runs to execute, by default 10
        base_seed : int, optional
            Base random seed, by default 42

        Returns
        -------
        Dict
            Dictionary containing sampling results including selected indices, diversity score, and algorithm metrics
        """
        print(f"\nRunning {self.method_name}")
        print(f"Algorithm: {self.algorithm.name}")
        print(f"Distance: {self.distance_calculator.name}")
        print(f"Total configurations: {len(self.physics_parameters)}")

        # Handle single run case (for parallel execution)
        # Handle single run case (for parallel execution)
        if n_runs == 1:
            seed = base_seed if base_seed else None
            indices, quality = self._run_single_sample(n_samples, seed)
            diversity = self.calculate_diversity_score_generic(indices, self.distance_calculator.name)

            result = {
                'selected_indices': indices,
                'diversity_score': diversity,
                'best_run': 1,
                'total_runs': 1,
                'all_diversities': [diversity],
                'algorithm': self.algorithm.name,
                'distance': self.distance_calculator.name,
                'selection_metric': 'inertia' if self.algorithm_name == 'kmedoids' else 'diversity'
            }

            # Add inertia for k-means
            if self.algorithm_name == 'kmedoids':
                result['inertia'] = quality

                # ADDED: Include cluster assignments if available
                if hasattr(self.algorithm, 'cluster_assignments'):
                    result['cluster_assignments'] = self.algorithm.cluster_assignments.tolist()
                    result['n_clusters'] = self.algorithm.n_clusters

            return result

        # Multiple runs - full logic
        best_indices = None
        best_score = -1 if self.algorithm_name == 'greedy' else float('inf')
        best_diversity = -1
        best_run = 0
        all_diversities = []

        # For k-means, track inertia
        run_stats = []
        all_inertia_scores = []

        for run in range(n_runs):
            seed = base_seed + run if base_seed else None

            # Run algorithm
            indices = self.algorithm.select_samples(
                self.distance_calculator,
                n_samples,
                len(self.physics_parameters),
                seed
            )

            # Calculate diversity
            diversity = self.calculate_diversity_score_generic(indices, self.distance_calculator.name)
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
                if self.algorithm_name == 'kmedoids':
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
        if self.algorithm_name == 'kmedoids' and run_stats:
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
            'selection_metric': 'inertia' if self.algorithm_name == 'kmedoids' else 'diversity'
        }

        # Add inertia for k-means
        if self.algorithm_name == 'kmedoids':
            result['inertia'] = best_score
            result['all_inertia_scores'] = all_inertia_scores

            # ADDED: Include cluster assignments from the best run
            # Need to re-run the best configuration to get cluster assignments
            best_seed = base_seed + (best_run - 1) if base_seed else None
            indices_check, _ = self._run_single_sample(n_samples, best_seed)

            if hasattr(self.algorithm, 'cluster_assignments'):
                result['cluster_assignments'] = self.algorithm.cluster_assignments.tolist()
                result['n_clusters'] = self.algorithm.n_clusters

        return result
