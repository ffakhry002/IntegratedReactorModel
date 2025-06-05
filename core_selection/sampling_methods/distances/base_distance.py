"""
Base distance abstract class for distance calculations.
"""
from abc import ABC, abstractmethod

class BaseDistance(ABC):
    """Abstract base class for distance calculations."""

    def __init__(self, name: str):
        self.name = name
        self.supports_feature_space = False

    @abstractmethod
    def get_distance(self, idx1: int, idx2: int) -> float:
        """Calculate distance between two items by their indices."""
        pass
