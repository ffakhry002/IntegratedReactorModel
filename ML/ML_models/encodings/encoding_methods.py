import numpy as np
from typing import List, Tuple, Dict
import networkx as nx
from scipy.spatial import distance_matrix

class ReactorEncodings:

    @staticmethod
    def one_hot_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """One-hot encoding with position features.

        Method 1: One-hot encoding with position features
        FIXED: All irradiation positions encoded identically - no label-specific features

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout

        Returns
        -------
        tuple
            (features, irr_positions, position_order) where features is the encoded array,
            irr_positions is list of irradiation positions, and position_order is sorted positions
        """
        feature_vec = []
        irr_positions = []

        # First pass: identify all irradiation positions
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))

        # Sort positions by row first, then column for consistent ordering
        position_order = sorted(irr_positions)

        # Cell type encoding - NO LABEL-SPECIFIC FEATURES
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                cell = lattice[i, j]
                if cell == 'F':
                    feature_vec.extend([1, 0, 0])  # Fuel
                elif cell == 'C':
                    feature_vec.extend([0, 1, 0])  # Control
                elif cell.startswith('I'):
                    # ALL IRRADIATION POSITIONS ENCODED THE SAME
                    feature_vec.extend([0, 0, 1])  # Just mark as irradiation
                else:
                    feature_vec.extend([0, 0, 0])  # Empty/Other

        # Position encoding - this provides the spatial information
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                # Normalized position features
                feature_vec.extend([i/7, j/7])

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def categorical_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """Simple categorical encoding.

        Method 2: Simple categorical encoding
        FIXED: All irradiation positions get the same categorical value

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout

        Returns
        -------
        tuple
            (features, irr_positions, position_order) where features is the encoded array,
            irr_positions is list of irradiation positions, and position_order is sorted positions
        """
        feature_vec = []
        irr_positions = []

        # First pass: identify irradiation positions
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))

        position_order = sorted(irr_positions)

        # Categorical encoding - ALL IRRADIATION POSITIONS GET VALUE 2
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                cell = lattice[i, j]
                if cell == 'C':
                    feature_vec.append(0)
                elif cell == 'F':
                    feature_vec.append(1)
                elif cell.startswith('I'):
                    # ALL irradiation positions get the same value
                    feature_vec.append(2)
                else:
                    feature_vec.append(-1)  # Unknown

        # Add radial distance for each cell - spatial feature
        center = 3.5
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                radial_dist = np.sqrt((i - center)**2 + (j - center)**2) / (center * np.sqrt(2))
                feature_vec.append(radial_dist)

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def physics_based_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """Physics-based encoding with global and local features.

        Method 3: Physics-based encoding with global and local features
        FIXED: No label-specific features, only spatial physics

        Returns feature vector with:
        - 2 global features (avg distance, symmetry balance)
        - 4 local features per irradiation position (fuel density, coolant contact, edge distance, center distance)
        - 1 NCI value per irradiation position
        Total: 2 + 4*(4+1) = 22 features for 4 positions

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout

        Returns
        -------
        tuple
            (features, irr_positions, position_order) where features is the encoded array,
            irr_positions is list of irradiation positions, and position_order is sorted positions
        """
        irr_positions = []

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))

        position_order = sorted(irr_positions)

        # Global features based on positions only
        global_features = ReactorEncodings._compute_global_features(irr_positions)

        # Local features for each irradiation position in spatial order
        local_features = []
        for pos in position_order:
            i, j = pos
            local_feat = ReactorEncodings._compute_local_features(lattice, i, j)
            # NO LABEL-SPECIFIC FEATURES - just spatial physics
            local_features.extend(local_feat)

        # Add NCI features for each position
        nci_features = ReactorEncodings._compute_nci_for_positions(position_order)

        # Combine all features
        feature_vec = np.concatenate([global_features, local_features, nci_features])

        return feature_vec, irr_positions, position_order

    @staticmethod
    def _compute_global_features(positions: List[Tuple]) -> np.ndarray:
        """Compute global configuration features using paper's coordinate system.

        Parameters
        ----------
        positions : List[Tuple]
            List of (row, col) tuples for irradiation positions

        Returns
        -------
        np.ndarray
            Array of global features
        """
        # Convert to continuous coordinates (center of cells)
        continuous_positions = [(i + 0.5, j + 0.5) for i, j in positions]
        positions_array = np.array(continuous_positions)

        # Reactor center is at (4, 4) in continuous coordinates
        center = np.array([4.0, 4.0])

        # 1. Average distance to core center
        distances = [np.linalg.norm(pos - center) for pos in positions_array]
        avg_distance = np.mean(distances)
        max_possible_dist = np.sqrt(2) * 4  # From center to corner
        avg_distance_norm = avg_distance / max_possible_dist

        # 2. Symmetry balance (center of mass distance from reactor center)
        center_of_mass = np.mean(positions_array, axis=0)
        symmetry_balance = np.linalg.norm(center_of_mass - center)
        symmetry_balance_norm = symmetry_balance / max_possible_dist

        return np.array([avg_distance_norm, symmetry_balance_norm])

    @staticmethod
    def _compute_local_features(lattice: np.ndarray, i: int, j: int) -> List[float]:
        """Compute local features for a specific position using paper's definitions.

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout
        i : int
            Row position
        j : int
            Column position

        Returns
        -------
        List[float]
            List of local features for the position
        """
        # Position center for distance calculations
        pos_center = (i + 0.5, j + 0.5)
        reactor_center = (4.0, 4.0)

        # 1. Local fuel density (adjacent fuel cells - 8 neighbors including diagonals)
        fuel_count = 0
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < 8 and 0 <= nj < 8:
                if lattice[ni, nj] == 'F':  # Only count F, not I
                    fuel_count += 1
        local_fuel_density = fuel_count / 8  # Max is 8

        # 2. Coolant contact count (number of adjacent coolant cells)
        # Special positions that should have coolant_contact_norm = 0.5
        special_positions = [(1, 2), (2, 1), (5, 1), (6, 2), (6, 5), (5, 6), (2, 6), (1, 5)]

        if (i, j) in special_positions:
            coolant_contact_norm = 0.0
        else:
            coolant_count = 0
            # Check only the 4 directly adjacent cells (not diagonal)
            adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

            for di, dj in adjacent_offsets:
                ni, nj = i + di, j + dj
                # Check if adjacent to actual coolant cell
                if 0 <= ni < 8 and 0 <= nj < 8:
                    if lattice[ni, nj] == 'C':
                        coolant_count += 1
                else:
                    # Outside grid boundary counts as coolant (virtual coolant)
                    coolant_count += 1

            # Normalize by max possible (2 in circular reactor design)
            coolant_contact_norm = coolant_count / 2.0

        # 3. Edge distance with special handling for corner positions
        # Check if in one of the four corner positions

        edge_dist = min(i, j, 7-i, 7-j)

        edge_dist_norm = edge_dist / 3.5  # Max is 3.5

        # 4. Distance to core center
        center_dist = np.sqrt((pos_center[0] - reactor_center[0])**2 +
                            (pos_center[1] - reactor_center[1])**2)
        max_dist = np.sqrt(2) * 4
        center_dist_norm = center_dist / max_dist

        return [local_fuel_density, coolant_contact_norm, edge_dist_norm, center_dist_norm]


    @staticmethod
    def _compute_nci_for_positions(positions: List[Tuple], lambda_decay: float = 1.5) -> List[float]:
        """Compute Neutron Competition Index for each irradiation position.

        NCI_i = sum over j≠i of contribution based on distance thresholds:
        - d < sqrt(4.9): exp(-d_ij / lambda) (exponential decay)
        - sqrt(4.9) <= d <= sqrt(5.1): 0.1 (constant small value)
        - d > sqrt(5.1): 0 (no contribution)

        Parameters
        ----------
        positions : List[Tuple]
            List of (row, col) tuples for irradiation positions
        lambda_decay : float, optional
            Decay parameter, by default 1.5

        Returns
        -------
        List[float]
            List of NCI values, one per position
        """
        # Convert to continuous coordinates (center of cells)
        continuous_positions = [(i + 0.5, j + 0.5) for i, j in positions]

        # Distance thresholds
        threshold_low = np.sqrt(4.9)   # ~2.21
        threshold_high = np.sqrt(5.1)  # ~2.26

        nci_values = []
        for i, pos_i in enumerate(continuous_positions):
            nci = 0.0
            for j, pos_j in enumerate(continuous_positions):
                if i != j:
                    # Euclidean distance
                    dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)

                    # Apply threshold-based contribution
                    if dist < threshold_low:
                        # Close distance: exponential decay
                        nci += np.exp(-dist / lambda_decay)
                    elif threshold_low <= dist <= threshold_high:
                        # Medium distance: constant small contribution
                        nci += 0.1
                    # else: dist > threshold_high, contributes 0 (no addition)

            nci_values.append(nci)

        return nci_values

    @staticmethod
    def spatial_convolution_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """Spatial convolution-like encoding.

        Method 4: Spatial convolution-like encoding
        FIXED: All irradiation positions encoded identically

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout

        Returns
        -------
        tuple
            (features, irr_positions, position_order) where features is the encoded array,
            irr_positions is list of irradiation positions, and position_order is sorted positions
        """
        feature_vec = []
        irr_positions = []

        # First pass: identify irradiation positions
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))

        position_order = sorted(irr_positions)

        # Pad lattice
        padded = np.pad(lattice, 1, mode='constant', constant_values='X')

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                # Get 3x3 neighborhood
                pi, pj = i + 1, j + 1
                neighborhood = padded[pi-1:pi+2, pj-1:pj+2].flatten()

                # Encode neighborhood - NO LABEL-SPECIFIC ENCODING
                for cell in neighborhood:
                    if cell == 'C':
                        feature_vec.extend([1, 0, 0, 0])
                    elif cell == 'F':
                        feature_vec.extend([0, 1, 0, 0])
                    elif cell.startswith('I'):
                        # ALL irradiation positions encoded the same
                        feature_vec.extend([0, 0, 1, 0])
                    else:  # padding 'X' or empty
                        feature_vec.extend([0, 0, 0, 1])

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def graph_based_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """Graph-based encoding.

        Method 5: Graph-based encoding
        FIXED: No label-specific features

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout

        Returns
        -------
        tuple
            (features, irr_positions, position_order) where features is the encoded array,
            irr_positions is list of irradiation positions, and position_order is sorted positions
        """
        G = nx.Graph()
        irr_positions = []
        node_features = {}

        # Get lattice dimensions dynamically
        n_rows, n_cols = lattice.shape

        # First pass: identify irradiation positions
        for i in range(n_rows):
            for j in range(n_cols):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))

        position_order = sorted(irr_positions)

        # Create nodes - NO LABEL-SPECIFIC FEATURES
        for i in range(n_rows):
            for j in range(n_cols):
                node_id = i * n_cols + j
                cell = lattice[i, j]

                # Node features - all irradiation positions get same encoding
                if cell == 'C':
                    features = [1, 0, 0]
                elif cell == 'F':
                    features = [0, 1, 0]
                elif cell.startswith('I'):
                    # ALL irradiation positions encoded the same
                    features = [0, 0, 1]
                else:
                    features = [0, 0, 0]

                node_features[node_id] = features
                G.add_node(node_id, features=features, pos=(i, j))

        # Create edges (adjacent cells)
        for i in range(n_rows):
            for j in range(n_cols):
                node_id = i * n_cols + j
                # Connect to adjacent cells
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_rows and 0 <= nj < n_cols:
                        neighbor_id = ni * n_cols + nj
                        G.add_edge(node_id, neighbor_id)

        # Extract features
        feature_vec = []
        total_nodes = n_rows * n_cols

        for node in range(total_nodes):
            # Own features
            feature_vec.extend(node_features[node])

            # Aggregate neighbor features
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_avg = np.mean([node_features[n] for n in neighbors], axis=0)
                feature_vec.extend(neighbor_avg)
            else:
                feature_vec.extend([0, 0, 0])

            # Add centrality measure
            degree_centrality = G.degree(node) / 4  # Max degree is still 4
            feature_vec.append(degree_centrality)

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def raw_2d_grid(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """Raw 2D grid for CNN input.

        Method 6: Raw 2D grid for CNN input
        FIXED: All irradiation positions in same channel

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout

        Returns
        -------
        tuple
            (grid, irr_positions, position_order) where grid is the 3D encoded array,
            irr_positions is list of irradiation positions, and position_order is sorted positions
        """
        irr_positions = []

        # Dynamic size based on lattice
        n_rows, n_cols = lattice.shape
        grid = np.zeros((n_rows, n_cols, 3))  # 3 channels: C, F, I (any irradiation)

        for i in range(n_rows):
            for j in range(n_cols):
                cell = lattice[i, j]
                if cell == 'C':
                    grid[i, j, 0] = 1
                elif cell == 'F':
                    grid[i, j, 1] = 1
                elif cell.startswith('I'):
                    # ALL irradiation positions in the same channel
                    grid[i, j, 2] = 1
                    irr_positions.append((i, j))

        position_order = sorted(irr_positions)
        return grid, irr_positions, position_order
