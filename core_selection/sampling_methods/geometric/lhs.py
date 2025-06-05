"""
Latin Hypercube Sampling module.
"""

import numpy as np
from scipy.stats import qmc
from typing import Dict, List, Tuple
from .base_qmc import BaseQMCSampler


class LHSSampler(BaseQMCSampler):
    """Latin Hypercube Sampling in physics-informed parameter space."""

    def __init__(self, use_6x6_restriction=False):
        self.method_name = "lhs"
        super().__init__(use_6x6_restriction=use_6x6_restriction)

    def sample(self, n_samples: int, n_runs: int = 10, base_seed: int = 42) -> Dict:
        """
        Perform Latin Hypercube Sampling multiple times and return the most diverse set.

        Args:
            n_samples: Number of samples to generate
            n_runs: Number of runs to find most diverse set
            base_seed: Base random seed

        Returns:
            Dictionary with sampling results
        """
        return self._run_multiple_times(
            n_samples, n_runs, base_seed,
            "LHS", self._single_lhs_run
        )

    def _single_lhs_run(self, n_samples: int, seed: int = None) -> Tuple[List[int], List[float]]:
        """Perform a single LHS run."""
        # Get parameter ranges
        n_features = self.feature_matrix.shape[1]
        param_ranges = self._get_parameter_ranges()

        # Generate LHS samples in [0, 1]^d
        sampler = qmc.LatinHypercube(d=n_features, seed=seed)
        lhs_samples_unit = sampler.random(n=n_samples)

        # Scale to actual parameter ranges
        lhs_samples = self._scale_samples_to_ranges(lhs_samples_unit, param_ranges)

        # Normalize the LHS samples using the same scaler
        lhs_samples_normalized = self.scaler.transform(lhs_samples)

        # Find closest discrete configurations
        return self._find_closest_configurations(lhs_samples_normalized, n_samples)
