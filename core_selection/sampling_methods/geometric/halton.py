"""
Halton sequence sampling module.
"""

import numpy as np
from scipy.stats import qmc
from typing import Dict, List, Tuple
from .base_qmc import BaseQMCSampler


class HaltonSampler(BaseQMCSampler):
    """Halton sequence sampling in physics-informed parameter space."""

    def __init__(self, use_6x6_restriction=False, selected_parameters=None):
        self.method_name = "halton"
        super().__init__(use_6x6_restriction=use_6x6_restriction,
                        selected_parameters=selected_parameters)

    def sample(self, n_samples: int, n_runs: int = 10, base_seed: int = 42) -> Dict:
        """
        Perform Halton sequence sampling with scrambling.

        Args:
            n_samples: Number of samples to generate
            n_runs: Number of runs to find most diverse set (with scrambling)
            base_seed: Base random seed

        Returns:
            Dictionary with sampling results
        """
        print("\nUsing scrambled Halton sequences for better coverage")
        print(f"Note: Halton sequences work well with any number of samples (no power-of-2 restriction)")

        results = self._run_multiple_times(
            n_samples, n_runs, base_seed,
            "Scrambled Halton", self._single_halton_run
        )

        # Add Halton-specific fields
        results['deterministic'] = False  # Using scrambling

        return results

    def _single_halton_run(self, n_samples: int, seed: int = None) -> Tuple[List[int], List[float]]:
        """Perform a single Halton sampling run with the given seed."""
        # Get parameter ranges
        n_features = self.feature_matrix.shape[1]
        param_ranges = self._get_parameter_ranges()

        # Generate Halton sequence in [0, 1]^d with scrambling and seed
        sampler = qmc.Halton(d=n_features, scramble=True, seed=seed)
        halton_samples_unit = sampler.random(n=n_samples)

        # Scale to actual parameter ranges
        halton_samples = self._scale_samples_to_ranges(halton_samples_unit, param_ranges)

        # Normalize the Halton samples
        halton_samples_normalized = self.scaler.transform(halton_samples)

        # Find closest discrete configurations
        return self._find_closest_configurations(halton_samples_normalized, n_samples)
