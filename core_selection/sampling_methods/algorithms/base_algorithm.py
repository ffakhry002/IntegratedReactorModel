"""
Base algorithm abstract class for sampling algorithms.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple

class BaseAlgorithm(ABC):
    """Abstract base class for sampling algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_samples(self, distance_calculator, n_samples: int,
                      n_items: int, seed: Optional[int] = None) -> List[int]:
        """
        Select n_samples from n_items using the provided distance calculator.

        Args:
            distance_calculator: Object with get_distance(i, j) method
            n_samples: Number of samples to select
            n_items: Total number of items to choose from
            seed: Random seed for reproducibility

        Returns:
            List of selected indices
        """
        pass

    def supports_quality_metric(self) -> bool:
        """Whether this algorithm has its own quality metric."""
        return False

    def get_quality_metric(self) -> Tuple[Optional[float], Optional[str]]:
        """Return (metric_value, metric_name) if available."""
        return None, None
