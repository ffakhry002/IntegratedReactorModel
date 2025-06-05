"""
Unified discretization for consistent Jaccard calculations across samplers.
"""
import numpy as np


class UnifiedDiscretization:
    """Unified discretization method for consistent Jaccard calculations across all samplers."""

    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self._bin_edges_cache = None
        self._full_dataset_stats = None

    def fit(self, full_feature_matrix: np.ndarray):
        """Fit discretization parameters on the full dataset."""
        self._full_dataset_stats = {
            'means': np.mean(full_feature_matrix, axis=0),
            'stds': np.std(full_feature_matrix, axis=0),
            'mins': np.min(full_feature_matrix, axis=0),
            'maxs': np.max(full_feature_matrix, axis=0)
        }

        # Compute bin edges for each feature using the full dataset
        self._bin_edges_cache = []
        for i in range(full_feature_matrix.shape[1]):
            col_data = full_feature_matrix[:, i]

            # Use quantile-based binning for better distribution
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(col_data, quantiles)

            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)

            # Handle edge case of too few unique values
            if len(bin_edges) <= 2:
                # Create artificial bins around the range
                min_val, max_val = np.min(col_data), np.max(col_data)
                if min_val == max_val:
                    # All values identical - create small range
                    range_ext = 0.1 * (abs(min_val) + 1e-6)
                    bin_edges = np.linspace(min_val - range_ext,
                                          min_val + range_ext,
                                          self.n_bins + 1)
                else:
                    # Extend range slightly for numerical stability
                    range_val = max_val - min_val
                    bin_edges = np.linspace(min_val - 0.01 * range_val,
                                          max_val + 0.01 * range_val,
                                          self.n_bins + 1)

            self._bin_edges_cache.append(bin_edges)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features to discretized form using consistent binning."""
        if self._bin_edges_cache is None:
            raise ValueError("Must call fit() before transform()")

        discretized = np.zeros_like(features, dtype=np.int8)

        for i in range(features.shape[1]):
            bin_edges = self._bin_edges_cache[i]
            col_data = features[:, i]

            # Clip to ensure values are within bin range
            col_data_clipped = np.clip(col_data, bin_edges[0], bin_edges[-1] - 1e-15)

            # Assign to bins (subtract 1 to get 0-based indexing)
            discretized[:, i] = np.digitize(col_data_clipped, bin_edges[1:-1])

        return discretized

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
