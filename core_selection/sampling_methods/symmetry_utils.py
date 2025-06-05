"""
Symmetry utilities for 8x8 lattice transformations.
Handles all 8 symmetries: identity, 3 rotations, and 4 reflections.
"""

import numpy as np
from typing import List, Tuple, Set


def rotate_90(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Rotate positions 90 degrees clockwise around the center of 8x8 grid."""
    return [(7 - y, x) for x, y in positions]


def rotate_180(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Rotate positions 180 degrees around the center of 8x8 grid."""
    return [(7 - x, 7 - y) for x, y in positions]


def rotate_270(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Rotate positions 270 degrees clockwise around the center of 8x8 grid."""
    return [(y, 7 - x) for x, y in positions]


def flip_horizontal(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Flip positions horizontally (about vertical axis through center)."""
    return [(7 - x, y) for x, y in positions]


def flip_vertical(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Flip positions vertically (about horizontal axis through center)."""
    return [(x, 7 - y) for x, y in positions]


def flip_diagonal_main(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Flip positions about main diagonal (top-left to bottom-right)."""
    return [(y, x) for x, y in positions]


def flip_diagonal_anti(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Flip positions about anti-diagonal (top-right to bottom-left)."""
    return [(7 - y, 7 - x) for x, y in positions]


def identity(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return positions unchanged."""
    return positions.copy()


# All 8 symmetry operations
ALL_SYMMETRIES = [
    identity,
    rotate_90,
    rotate_180,
    rotate_270,
    flip_horizontal,
    flip_vertical,
    flip_diagonal_main,
    flip_diagonal_anti
]


def apply_symmetry_to_4d_point(point: np.ndarray, symmetry_func) -> np.ndarray:
    """
    Apply symmetry transformation to a 4D continuous point.
    The point represents 4 positions in normalized [0,1] space.

    Fixed version that properly maps 4D to positions and back.
    """
    # Convert 4D point to 4 position pairs using the same mapping as lattice samplers
    positions = []

    # Use the corrected mapping (same as in the fixed lattice samplers)
    positions.append((int(point[0] * 8), int(point[1] * 8)))
    positions.append((int(point[2] * 8), int(point[3] * 8)))
    positions.append((int(point[1] * 8), int(point[2] * 8)))
    positions.append((int(point[3] * 8), int(point[0] * 8)))

    # Apply symmetry transformation
    transformed_positions = symmetry_func(positions)

    # Convert back to 4D point
    # This is the tricky part - we need to find a 4D point that maps to these positions
    # We'll use the first 4 coordinates directly (simplified approach)
    result = np.zeros(4)

    # Map back (using first occurrence of each dimension)
    # This is an approximation since the mapping isn't bijective
    result[0] = (transformed_positions[0][0] + 0.5) / 8.0
    result[1] = (transformed_positions[0][1] + 0.5) / 8.0
    result[2] = (transformed_positions[1][0] + 0.5) / 8.0
    result[3] = (transformed_positions[1][1] + 0.5) / 8.0

    return result

def are_configurations_symmetric(positions1: List[Tuple[int, int]],
                               positions2: List[Tuple[int, int]]) -> bool:
    """Check if two configurations are symmetric versions of each other."""
    set1 = set(positions1)

    for symmetry in ALL_SYMMETRIES:
        set2_transformed = set(symmetry(positions2))
        if set1 == set2_transformed:
            return True

    return False


def get_canonical_form(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Get the canonical (normalized) form of a configuration.
    Uses lexicographic ordering to ensure unique representation.
    """
    all_forms = []

    for symmetry in ALL_SYMMETRIES:
        transformed = sorted(symmetry(positions))
        all_forms.append(transformed)

    # Return the lexicographically smallest form
    return min(all_forms)
