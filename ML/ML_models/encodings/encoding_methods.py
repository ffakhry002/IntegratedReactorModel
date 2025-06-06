import numpy as np
from typing import List, Tuple, Dict
import networkx as nx
from scipy.spatial import distance_matrix

class ReactorEncodings:

    @staticmethod
    def one_hot_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Method 1: One-hot encoding with position features
        Now also encodes WHICH irradiation position (I_1, I_2, etc.) is at each location
        Returns: (features, irr_positions, position_order)
        """
        feature_vec = []
        irr_positions = []
        irr_labels = {}  # Map position to label

        # First pass: identify irradiation positions and their labels
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                cell = lattice[i, j]
                if cell.startswith('I'):
                    irr_positions.append((i, j))
                    irr_labels[(i, j)] = cell

        # Sort positions by row first, then column for consistent ordering
        position_order = sorted(irr_positions)

        # Cell type encoding with extended features for irradiation positions
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                cell = lattice[i, j]
                if cell == 'F':
                    feature_vec.extend([1, 0, 0, 0, 0, 0, 0])  # Fuel
                elif cell == 'C':
                    feature_vec.extend([0, 1, 0, 0, 0, 0, 0])  # Control
                elif cell.startswith('I'):
                    # Extended encoding for irradiation positions
                    if cell == 'I_1':
                        feature_vec.extend([0, 0, 1, 1, 0, 0, 0])
                    elif cell == 'I_2':
                        feature_vec.extend([0, 0, 1, 0, 1, 0, 0])
                    elif cell == 'I_3':
                        feature_vec.extend([0, 0, 1, 0, 0, 1, 0])
                    elif cell == 'I_4':
                        feature_vec.extend([0, 0, 1, 0, 0, 0, 1])
                    else:
                        feature_vec.extend([0, 0, 1, 0, 0, 0, 0])  # Unknown I
                else:
                    feature_vec.extend([0, 0, 0, 0, 0, 0, 0])

        # Position encoding
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                feature_vec.extend([i/7, j/7])

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def categorical_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Method 2: Simple categorical encoding
        Enhanced to distinguish between different irradiation positions
        """
        feature_vec = []
        irr_positions = []

        # First pass: identify irradiation positions
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))

        position_order = sorted(irr_positions)

        # Categorical encoding with distinct values for each I_x
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                cell = lattice[i, j]
                if cell == 'C':
                    feature_vec.append(0)
                elif cell == 'F':
                    feature_vec.append(1)
                elif cell == 'I_1':
                    feature_vec.append(2)
                elif cell == 'I_2':
                    feature_vec.append(3)
                elif cell == 'I_3':
                    feature_vec.append(4)
                elif cell == 'I_4':
                    feature_vec.append(5)
                else:
                    feature_vec.append(-1)  # Unknown

        # Add radial distance for each cell
        center = 3.5
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                radial_dist = np.sqrt((i - center)**2 + (j - center)**2) / (center * np.sqrt(2))
                feature_vec.append(radial_dist)

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def physics_based_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Method 3: Physics-based encoding with global and local features
        Enhanced to include irradiation position identity
        """
        irr_positions = []
        irr_labels = {}

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))
                    irr_labels[(i, j)] = lattice[i, j]

        position_order = sorted(irr_positions)

        # Global features
        global_features = ReactorEncodings._compute_global_features(irr_positions)

        # Local features for each irradiation position in spatial order
        local_features = []
        for pos in position_order:
            i, j = pos
            local_feat = ReactorEncodings._compute_local_features(lattice, i, j)

            # Add irradiation position identity features
            label = irr_labels[pos]
            if label == 'I_1':
                local_feat.extend([1, 0, 0, 0])
            elif label == 'I_2':
                local_feat.extend([0, 1, 0, 0])
            elif label == 'I_3':
                local_feat.extend([0, 0, 1, 0])
            elif label == 'I_4':
                local_feat.extend([0, 0, 0, 1])
            else:
                local_feat.extend([0, 0, 0, 0])

            local_features.extend(local_feat)

        # Combine
        feature_vec = np.concatenate([global_features, local_features])

        return feature_vec, irr_positions, position_order

    @staticmethod
    def _compute_global_features(positions: List[Tuple]) -> np.ndarray:
        """Compute global configuration features using paper's coordinate system"""
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

        # 3. Clustering coefficient (minimum enclosing circle radius)
        if len(positions) > 1:
            # Use the approach from the paper
            centroid = np.mean(positions_array, axis=0)
            max_dist_from_centroid = max([np.linalg.norm(pos - centroid) for pos in positions_array])
            clustering_coeff = max_dist_from_centroid / max_possible_dist
        else:
            clustering_coeff = 0

        return np.array([avg_distance_norm, symmetry_balance_norm, clustering_coeff])

    @staticmethod
    def _compute_local_features(lattice: np.ndarray, i: int, j: int) -> List[float]:
        """Compute local features for a specific position using paper's definitions"""
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

        # 2. Distance to edge (minimum distance to outer rim)
        edge_dist = min(i, j, 7-i, 7-j)
        edge_dist_norm = edge_dist / 3.5  # Max is 3.5

        # 3. Distance to core center
        center_dist = np.sqrt((pos_center[0] - reactor_center[0])**2 +
                            (pos_center[1] - reactor_center[1])**2)
        max_dist = np.sqrt(2) * 4
        center_dist_norm = center_dist / max_dist

        return [local_fuel_density, edge_dist_norm, center_dist_norm]

    @staticmethod
    def spatial_convolution_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Method 4: Spatial convolution-like encoding
        Enhanced to distinguish between irradiation positions
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

                # Encode neighborhood with distinct irradiation encodings
                for cell in neighborhood:
                    if cell == 'C':
                        feature_vec.extend([1, 0, 0, 0, 0, 0, 0, 0])
                    elif cell == 'F':
                        feature_vec.extend([0, 1, 0, 0, 0, 0, 0, 0])
                    elif cell == 'I_1':
                        feature_vec.extend([0, 0, 1, 1, 0, 0, 0, 0])
                    elif cell == 'I_2':
                        feature_vec.extend([0, 0, 1, 0, 1, 0, 0, 0])
                    elif cell == 'I_3':
                        feature_vec.extend([0, 0, 1, 0, 0, 1, 0, 0])
                    elif cell == 'I_4':
                        feature_vec.extend([0, 0, 1, 0, 0, 0, 1, 0])
                    elif cell.startswith('I'):
                        feature_vec.extend([0, 0, 1, 0, 0, 0, 0, 0])
                    else:  # padding 'X'
                        feature_vec.extend([0, 0, 0, 0, 0, 0, 0, 1])

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def graph_based_encoding(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Method 5: Graph-based encoding
        Enhanced to include irradiation position identity
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

        # Create nodes with enhanced features
        for i in range(n_rows):
            for j in range(n_cols):
                node_id = i * n_cols + j
                cell = lattice[i, j]

                # Node features with irradiation position identity
                if cell == 'C':
                    features = [1, 0, 0, 0, 0, 0, 0]
                elif cell == 'F':
                    features = [0, 1, 0, 0, 0, 0, 0]
                elif cell == 'I_1':
                    features = [0, 0, 1, 1, 0, 0, 0]
                elif cell == 'I_2':
                    features = [0, 0, 1, 0, 1, 0, 0]
                elif cell == 'I_3':
                    features = [0, 0, 1, 0, 0, 1, 0]
                elif cell == 'I_4':
                    features = [0, 0, 1, 0, 0, 0, 1]
                elif cell.startswith('I'):
                    features = [0, 0, 1, 0, 0, 0, 0]
                else:
                    features = [0, 0, 0, 0, 0, 0, 0]

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
                feature_vec.extend([0, 0, 0, 0, 0, 0, 0])

            # Add centrality measure
            degree_centrality = G.degree(node) / 4  # Max degree is still 4
            feature_vec.append(degree_centrality)

        return np.array(feature_vec), irr_positions, position_order

    @staticmethod
    def raw_2d_grid(lattice: np.ndarray) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Method 6: Raw 2D grid for CNN input
        Enhanced to distinguish irradiation positions
        """
        irr_positions = []

        # Dynamic size based on lattice
        n_rows, n_cols = lattice.shape
        grid = np.zeros((n_rows, n_cols, 7))  # 7 channels for C, F, I, I_1, I_2, I_3, I_4

        for i in range(n_rows):
            for j in range(n_cols):
                cell = lattice[i, j]
                if cell == 'C':
                    grid[i, j, 0] = 1
                elif cell == 'F':
                    grid[i, j, 1] = 1
                elif cell == 'I_1':
                    grid[i, j, 2] = 1
                    grid[i, j, 3] = 1
                    irr_positions.append((i, j))
                elif cell == 'I_2':
                    grid[i, j, 2] = 1
                    grid[i, j, 4] = 1
                    irr_positions.append((i, j))
                elif cell == 'I_3':
                    grid[i, j, 2] = 1
                    grid[i, j, 5] = 1
                    irr_positions.append((i, j))
                elif cell == 'I_4':
                    grid[i, j, 2] = 1
                    grid[i, j, 6] = 1
                    irr_positions.append((i, j))
                elif cell.startswith('I'):
                    grid[i, j, 2] = 1
                    irr_positions.append((i, j))

        position_order = sorted(irr_positions)
        return grid, irr_positions, position_order
