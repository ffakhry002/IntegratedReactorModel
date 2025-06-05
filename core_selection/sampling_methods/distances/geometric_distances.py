"""
Geometric distance calculations for 5D parameter space.
"""
import numpy as np
from .base_distance import BaseDistance

class GeometricEuclidean(BaseDistance):
    """Euclidean distance in 5D physics parameter space."""

    def __init__(self, base_sampler):
        super().__init__("euclidean")
        self.sampler = base_sampler
        self.supports_feature_space = True
        self.feature_matrix = base_sampler.feature_matrix_normalized
        self.sklearn_metric = 'euclidean'

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Use cached Euclidean distance from base sampler."""
        return self.sampler.get_cached_euclidean_distance(idx1, idx2)

    def get_distance_to_point(self, point: np.ndarray, idx: int) -> float:
        """Calculate Euclidean distance from arbitrary point to data point at idx."""
        return np.linalg.norm(self.feature_matrix[idx] - point)


class GeometricManhattan(BaseDistance):
    """Manhattan (L1) distance in 5D physics parameter space."""

    def __init__(self, base_sampler):
        super().__init__("manhattan")
        self.sampler = base_sampler
        self.supports_feature_space = True
        self.feature_matrix = base_sampler.feature_matrix_normalized
        self.sklearn_metric = 'manhattan'

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Use cached Manhattan distance from base sampler."""
        return self.sampler.get_cached_manhattan_distance(idx1, idx2)

    def get_distance_to_point(self, point: np.ndarray, idx: int) -> float:
        """Calculate Manhattan distance from arbitrary point to data point at idx."""
        return np.sum(np.abs(self.feature_matrix[idx] - point))


class GeometricJaccard(BaseDistance):
    """Jaccard distance in 5D physics parameter space using continuous formulation."""

    def __init__(self, base_sampler):
        super().__init__("jaccard")
        self.sampler = base_sampler
        self.supports_feature_space = True
        self.feature_matrix = base_sampler.feature_matrix_normalized

        # No need for discretization anymore!
        self.discretized_features = None

        # Still support sklearn-style metric if needed
        self.sklearn_metric = self._continuous_jaccard_metric

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Use continuous Jaccard distance calculation."""
        feat1 = self.feature_matrix[idx1]
        feat2 = self.feature_matrix[idx2]

        # Ensure features are non-negative by shifting
        min_val = min(feat1.min(), feat2.min(), 0)
        feat1_shifted = feat1 - min_val
        feat2_shifted = feat2 - min_val

        # Continuous Jaccard calculation
        numerator = np.sum(np.minimum(feat1_shifted, feat2_shifted))
        denominator = np.sum(np.maximum(feat1_shifted, feat2_shifted))

        if denominator == 0:
            return 0.0

        jaccard_similarity = numerator / denominator
        return 1.0 - jaccard_similarity

    def get_distance_to_point(self, point: np.ndarray, idx: int) -> float:
        """Calculate continuous Jaccard distance from arbitrary point to data point."""
        feat_data = self.feature_matrix[idx]

        # Ensure non-negative
        min_val = min(point.min(), feat_data.min(), 0)
        point_shifted = point - min_val
        feat_shifted = feat_data - min_val

        numerator = np.sum(np.minimum(point_shifted, feat_shifted))
        denominator = np.sum(np.maximum(point_shifted, feat_shifted))

        if denominator == 0:
            return 0.0

        return 1.0 - (numerator / denominator)

    def _continuous_jaccard_metric(self, x, y):
        """Continuous Jaccard metric for sklearn compatibility."""
        # Ensure non-negative
        min_val = min(x.min(), y.min(), 0)
        x_shifted = x - min_val
        y_shifted = y - min_val

        numerator = np.sum(np.minimum(x_shifted, y_shifted))
        denominator = np.sum(np.maximum(x_shifted, y_shifted))

        if denominator == 0:
            return 0.0

        return 1.0 - (numerator / denominator)
