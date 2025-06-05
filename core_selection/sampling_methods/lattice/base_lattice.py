"""
Base class for lattice-based samplers with common functionality.
Place this file in the same directory as the lattice samplers.
"""

import numpy as np
from typing import List, Tuple
from ..base import BaseSampler
from ..symmetry_utils import ALL_SYMMETRIES


class BaseLatticeSampler(BaseSampler):
    """Base class for lattice-based samplers with common functionality."""

    def __init__(self, use_6x6_restriction=False):
        super().__init__(use_6x6_restriction=use_6x6_restriction)

        # Define forbidden coolant positions
        self.forbidden_positions = {
            (0, 0), (0, 7), (7, 0), (7, 7),  # Corners
            (0, 1), (1, 0),                   # Top-left adjacents
            (0, 6), (1, 7),                   # Top-right adjacents
            (6, 0), (7, 1),                   # Bottom-left adjacents
            (6, 7), (7, 6),                   # Bottom-right adjacents
        }

    def _continuous_to_valid_position(self, x: float, y: float) -> Tuple[int, int]:
        """
        Map a continuous point (x,y) in [0,1] × [0,1] to the nearest valid fuel position.
        Avoids forbidden coolant positions.
        """
        # Convert to 8×8 grid coordinates
        grid_x = x * 8
        grid_y = y * 8

        # Find the nearest valid position
        min_dist = float('inf')
        best_pos = None

        # Start with the cell containing the point
        base_x = int(grid_x)
        base_y = int(grid_y)

        # Ensure within bounds
        base_x = min(max(base_x, 0), 7)
        base_y = min(max(base_y, 0), 7)

        # If the base position is valid, use it
        if (base_x, base_y) not in self.forbidden_positions:
            return (base_x, base_y)

        # Otherwise, search in expanding rings for nearest valid position
        for radius in range(1, 8):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    # Only check perimeter of the square
                    if abs(di) == radius or abs(dj) == radius:
                        i = base_x + di
                        j = base_y + dj

                        if 0 <= i < 8 and 0 <= j < 8:
                            if (i, j) not in self.forbidden_positions:
                                # Calculate distance to continuous point
                                dist = ((i + 0.5) - grid_x)**2 + ((j + 0.5) - grid_y)**2
                                if dist < min_dist:
                                    min_dist = dist
                                    best_pos = (i, j)

            # If we found a valid position at this radius, return it
            if best_pos is not None:
                return best_pos

        # Fallback (should never reach here with 12 forbidden out of 64 positions)
        return (2, 2)  # A safe central position

    def _find_best_symmetry_match(self, continuous_positions: List[Tuple[int, int]],
                                 used_configs: set) -> int:
        """Find the best matching configuration considering all 8 symmetries."""
        best_config_idx = None
        min_total_distance = float('inf')

        # Convert continuous positions to coordinates
        continuous_coords = np.array([(x + 0.5, y + 0.5) for x, y in continuous_positions])

        for config_idx in range(len(self.irradiation_sets)):
            if config_idx in used_configs:
                continue

            # Try all 8 symmetries of the continuous positions
            for symmetry in ALL_SYMMETRIES:
                # Apply symmetry to continuous positions
                rotated_positions = symmetry(continuous_positions)
                rotated_coords = np.array([(x + 0.5, y + 0.5) for x, y in rotated_positions])

                # Get config coordinates
                config_coords = self.coords_cache[config_idx]

                # Calculate total distance using greedy matching
                total_distance = self._calculate_total_distance(rotated_coords, config_coords)

                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_config_idx = config_idx

        return best_config_idx

    def _calculate_total_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate total distance between two sets of coordinates using greedy matching."""
        distances = []
        used_indices = set()

        for coord1 in coords1:
            min_dist = float('inf')
            best_idx = -1

            for idx, coord2 in enumerate(coords2):
                if idx not in used_indices:
                    dist = np.linalg.norm(coord1 - coord2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx

            if best_idx != -1:
                distances.append(min_dist)
                used_indices.add(best_idx)

        return sum(distances) if distances else float('inf')
