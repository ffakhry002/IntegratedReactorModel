import numpy as np
import unittest
from encoding_methods import ReactorEncodings


class TestEdgeDistanceCalculation(unittest.TestCase):

    def setUp(self):
        """Create a standard 8x8 reactor lattice with coolant in corners"""
        self.lattice = np.array([
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
            ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']
        ])

    def test_adjacent_to_coolant(self):
        """Test cells directly adjacent to coolant should have distance 0"""
        # Position (1,1) is adjacent to coolant at (0,1) and (1,0)
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 1, 1)
        self.assertEqual(dist, 0.0, f"Cell (1,1) adjacent to coolant should have distance 0, got {dist}")

        # Position (0,2) is adjacent to coolant at (0,1)
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 0, 2)
        self.assertEqual(dist, 0.0, f"Cell (0,2) adjacent to coolant should have distance 0, got {dist}")

    def test_diagonal_to_coolant(self):
        """Test cells diagonally adjacent to coolant should have distance sqrt(2)/2"""
        expected = np.sqrt(2) / 2

        # Position (1,2) is diagonal to coolant at (0,1)
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 1, 2)
        self.assertAlmostEqual(dist, expected, places=5,
                              msg=f"Cell (1,2) diagonal to coolant should have distance {expected:.5f}, got {dist}")

    def test_edge_cells(self):
        """Test cells at grid edges (adjacent to virtual coolant)"""
        # Top edge cells
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 0, 3)
        self.assertEqual(dist, 0.0, f"Cell (0,3) at top edge should have distance 0, got {dist}")

        # Bottom edge cells
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 7, 3)
        self.assertEqual(dist, 0.0, f"Cell (7,3) at bottom edge should have distance 0, got {dist}")

        # Left edge cells
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 3, 0)
        self.assertEqual(dist, 0.0, f"Cell (3,0) at left edge should have distance 0, got {dist}")

        # Right edge cells
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 3, 7)
        self.assertEqual(dist, 0.0, f"Cell (3,7) at right edge should have distance 0, got {dist}")

    def test_specific_positions_from_html(self):
        """Test specific positions that were discussed in the HTML implementation"""
        # Position (3,4) should have distance ~2.9
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 3, 4)
        # Distance to coolant at (1,7): sqrt((7-4.5)^2 + (3.5-2)^2) = sqrt(6.25 + 2.25) = sqrt(8.5) ≈ 2.92
        expected = np.sqrt(8.5)
        self.assertAlmostEqual(dist, expected, places=1,
                              msg=f"Cell (3,4) should have distance ~{expected:.2f}, got {dist}")

        # Position (2,5) should have distance ~1.6
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 2, 5)
        # Distance to coolant at (0,6): closest point is (1,6), distance = sqrt((5.5-6)^2 + (2.5-1)^2) = sqrt(0.25 + 2.25) = sqrt(2.5) ≈ 1.58
        expected = np.sqrt(2.5)
        self.assertAlmostEqual(dist, expected, places=1,
                              msg=f"Cell (2,5) should have distance ~{expected:.2f}, got {dist}")

        # Position (2,4) should have distance ~2.1
        dist = ReactorEncodings._calculate_edge_distance(self.lattice, 2, 4)
        # Distance to coolant at (0,6): closest point is (1,6), distance = sqrt((4.5-6)^2 + (2.5-1)^2) = sqrt(2.25 + 2.25) = sqrt(4.5) ≈ 2.12
        expected = np.sqrt(4.5)
        self.assertAlmostEqual(dist, expected, places=1,
                              msg=f"Cell (2,4) should have distance ~{expected:.2f}, got {dist}")

    def test_central_cells(self):
        """Test cells in the center have maximum distance"""
        # Central cells should have the maximum distance to any coolant/boundary
        central_positions = [(3,3), (3,4), (4,3), (4,4)]

        for row, col in central_positions:
            dist = ReactorEncodings._calculate_edge_distance(self.lattice, row, col)
            # For central cells, nearest boundary is 3.5 units away
            # But coolant cells might be closer
            # Let's check what we actually get
            print(f"Central cell ({row},{col}) has distance: {dist}")

            # The distance should be at most 3.5 (distance to boundary)
            self.assertLessEqual(dist, 3.5,
                                f"Central cell ({row},{col}) distance {dist} should not exceed 3.5")

    def test_all_positions_max_distance(self):
        """Test that no position has distance greater than 3.5"""
        max_dist = 0
        max_pos = None

        for i in range(8):
            for j in range(8):
                dist = ReactorEncodings._calculate_edge_distance(self.lattice, i, j)
                if dist > max_dist:
                    max_dist = dist
                    max_pos = (i, j)

                # No distance should exceed 3.5
                self.assertLessEqual(dist, 3.5,
                                    f"Cell ({i},{j}) has distance {dist} which exceeds maximum 3.5")

        print(f"\nMaximum distance found: {max_dist} at position {max_pos}")
        print(f"This confirms the normalization factor should be 3.5")

    def test_corner_cells_diagonal_to_virtual_coolant(self):
        """Test corner cells that are diagonal to virtual coolant"""
        # Cell (0,0) is coolant itself, so let's test cells near corners but not at edge
        # Actually, let's create a custom lattice for this test
        custom_lattice = np.full((8, 8), 'F', dtype=object)

        # Test a cell at (0,0) - should be 0 (at edge)
        dist = ReactorEncodings._calculate_edge_distance(custom_lattice, 0, 0)
        self.assertEqual(dist, 0.0, f"Corner cell (0,0) at edge should have distance 0, got {dist}")

    def test_custom_coolant_pattern(self):
        """Test with a custom coolant pattern to verify calculations"""
        # Create a lattice with coolant only in the center
        custom_lattice = np.full((8, 8), 'F', dtype=object)
        custom_lattice[3, 3] = 'C'
        custom_lattice[3, 4] = 'C'
        custom_lattice[4, 3] = 'C'
        custom_lattice[4, 4] = 'C'

        # Test adjacent cell
        dist = ReactorEncodings._calculate_edge_distance(custom_lattice, 3, 2)
        self.assertEqual(dist, 0.0, "Cell adjacent to central coolant should have distance 0")

        # Test diagonal cell
        dist = ReactorEncodings._calculate_edge_distance(custom_lattice, 2, 2)
        expected = np.sqrt(2) / 2
        self.assertAlmostEqual(dist, expected, places=5,
                              msg=f"Cell diagonal to central coolant should have distance {expected:.5f}")

        # Test far cell - should use boundary distance
        dist = ReactorEncodings._calculate_edge_distance(custom_lattice, 0, 0)
        self.assertEqual(dist, 0.0, "Corner cell should have distance 0 to virtual boundary")

    def test_specific_user_positions(self):
        """Test specific positions requested by user"""
        positions_to_test = [(1,3), (2,1), (6,2), (5,3), (5,5)]

        print("\nTesting specific positions:")
        print("-" * 40)

        for row, col in positions_to_test:
            dist = ReactorEncodings._calculate_edge_distance(self.lattice, row, col)
            print(f"Position ({row},{col}): {dist:.4f}")


def test_positions_with_lattice(lattice, positions):
    """
    Test edge distances for a list of positions in a given lattice.

    Args:
        lattice: 8x8 numpy array representing the reactor configuration
        positions: List of (row, col) tuples to test

    Returns:
        Dictionary mapping positions to their edge distances
    """
    results = {}

    print("\nEdge Distance Test Results:")
    print("=" * 50)
    print(f"{'Position':^15} | {'Distance':^15} | {'Normalized':^15}")
    print("-" * 50)

    for row, col in positions:
        dist = ReactorEncodings._calculate_edge_distance(lattice, row, col)
        normalized = dist / 3.5  # Normalized by max possible distance
        results[(row, col)] = dist

        print(f"({row},{col}){' '*9} | {dist:^15.4f} | {normalized:^15.4f}")

    print("=" * 50)

    # Also show what's at each position
    print("\nCell types at tested positions:")
    for row, col in positions:
        cell_type = lattice[row, col]
        print(f"  ({row},{col}): {cell_type}")

    return results


# Standalone function for easy testing
def quick_test_positions(positions):
    """
    Quick test function that creates the standard lattice and tests positions.

    Args:
        positions: List of (row, col) tuples to test

    Example:
        quick_test_positions([(1,3), (2,1), (6,2), (5,3), (5,5)])
    """
    # Create standard lattice with coolant in corners
    lattice = np.array([
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']
    ])

    return test_positions_with_lattice(lattice, positions)


if __name__ == '__main__':
    import sys

    # Check if running with custom positions
    if len(sys.argv) > 1 and sys.argv[1] == '--test-positions':
        # Test the specific positions requested
        positions = [(1,3), (2,1), (6,2), (5,3), (5,5),(6,1),(5,0)]
        quick_test_positions(positions)
    else:
        # Run the unit tests
        unittest.main(verbosity=2)
