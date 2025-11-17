import numpy as np
from typing import List, Tuple, Dict
import networkx as nx
from scipy.spatial import distance_matrix
import os

# =============================================================================
# MANUAL CONFIGURATION TOGGLES - Change these to switch modes
# =============================================================================
IRRADIATION_MODE = 'fill'  # Options: 'vacuum' or 'fill'
                             # 'vacuum': I_1, I_2, I_3, I_4 (original)
                             # 'fill': I_1P, I_1B, I_1G, etc. (variable fill)

NCI_MODE = 'separate'          # Options: 'single' or 'separate'
                             # 'single': One NCI per position (standard)
                             # 'separate': NCI_P, NCI_B, NCI_G per position
                             # (only applies when IRRADIATION_MODE = 'fill')

# NCI Distance Cutoff - Read from environment variable, default to 1
# Set in SLURM script: export NCI_DISTANCE_CUTOFF=0, 1, or 2
NCI_DISTANCE_CUTOFF = int(os.environ.get('NCI_DISTANCE_CUTOFF', '1'))
                             # 0: No cutoff (all distances use exp(-(d-1)/lambda))
                             # 1: With cutoff (d > sqrt(5) → NCI = 0) [DEFAULT]
                             # 2: Don't use NCI features at all (only global + local)
# =============================================================================

class ReactorEncodings:

    @staticmethod
    def one_hot_encoding(lattice: np.ndarray, irradiation_mode: str = 'vacuum') -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
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
    def categorical_encoding(lattice: np.ndarray, irradiation_mode: str = 'vacuum') -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
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
    def physics_based_encoding(lattice: np.ndarray, irradiation_mode: str = None,
                               nci_mode: str = None, lambda_decay: float = 1.5,
                               lambda_P: float = 1.5, lambda_B: float = 1.5,
                               lambda_G: float = 1.5) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """Physics-based encoding with global and local features.

        Method 3: Physics-based encoding with global and local features
        Supports both vacuum (I_1, I_2) and fill (I_1P, I_1B, I_1G) modes.

        Vacuum mode features:
        - 2 global features (avg distance, symmetry balance)
        - 4 local features per position (fuel density, coolant contact, edge distance, center distance)
        - 1 NCI value per position
        Total: 2 + 4*4 + 4 = 22 features for 4 positions

        Fill mode features:
        - 5 global features (avg distance, symmetry balance, fraction_P, fraction_B, fraction_G)
        - 5 local features per position (fuel density, coolant contact, edge distance, center distance, vehicle_type)
        - 1 NCI per position (single mode) OR 3 NCI per position (separate mode)
        Total (single NCI): 5 + 4*5 + 4 = 29 features for 4 positions
        Total (separate NCI): 5 + 4*5 + 4*3 = 37 features for 4 positions

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout
        irradiation_mode : str
            'vacuum' for I_1, I_2 format or 'fill' for I_1P, I_1B, I_1G format
        nci_mode : str
            'single' for one NCI per position or 'separate' for NCI_P, NCI_B, NCI_G
        lambda_decay : float, optional
            Decay parameter for single NCI mode, by default 1.5
        lambda_P : float, optional
            Decay parameter for P-type vehicles (PWR) in separate mode, by default 1.5
        lambda_B : float, optional
            Decay parameter for B-type vehicles (BWR) in separate mode, by default 1.5
        lambda_G : float, optional
            Decay parameter for G-type vehicles (GAS) in separate mode, by default 1.5

        Returns
        -------
        tuple
            (features, irr_positions, position_order) where features is the encoded array,
            irr_positions is list of irradiation positions, and position_order is sorted positions
        """
        # Use module-level toggles if not explicitly provided
        if irradiation_mode is None:
            irradiation_mode = IRRADIATION_MODE
        if nci_mode is None:
            nci_mode = NCI_MODE

        irr_positions = []
        irr_positions_with_labels = []  # Store (i, j, label) for fill mode

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i, j].startswith('I'):
                    irr_positions.append((i, j))
                    irr_positions_with_labels.append((i, j, lattice[i, j]))

        position_order = sorted(irr_positions)
        position_order_with_labels = sorted(irr_positions_with_labels, key=lambda x: (x[0], x[1]))

        # Global features based on positions only
        global_features = ReactorEncodings._compute_global_features(
            irr_positions, position_order_with_labels, irradiation_mode)

        # Local features for each irradiation position in spatial order
        local_features = []
        for pos_data in position_order_with_labels:
            i, j = pos_data[0], pos_data[1]
            label = pos_data[2] if len(pos_data) > 2 else None
            local_feat = ReactorEncodings._compute_local_features(
                lattice, i, j, label, irradiation_mode)
            local_features.extend(local_feat)

        # Add NCI features for each position with lambda parameters (if enabled)
        if NCI_DISTANCE_CUTOFF == 2:
            # Skip NCI features entirely
            feature_vec = np.concatenate([global_features, local_features])
        else:
            # Include NCI features
            if irradiation_mode == 'fill' and nci_mode == 'separate':
                nci_features = ReactorEncodings._compute_nci_separate(
                    position_order_with_labels, lambda_P, lambda_B, lambda_G)
            else:
                nci_features = ReactorEncodings._compute_nci_for_positions(
                    position_order, lambda_decay)

            # Combine all features
            feature_vec = np.concatenate([global_features, local_features, nci_features])

        return feature_vec, irr_positions, position_order

    @staticmethod
    def _compute_global_features(positions: List[Tuple], positions_with_labels: List[Tuple],
                                 irradiation_mode: str) -> np.ndarray:
        """Compute global configuration features using paper's coordinate system.

        Parameters
        ----------
        positions : List[Tuple]
            List of (row, col) tuples for irradiation positions
        positions_with_labels : List[Tuple]
            List of (row, col, label) tuples for irradiation positions with labels
        irradiation_mode : str
            'vacuum' or 'fill'

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

        # For vacuum mode, return just these 2 features
        if irradiation_mode == 'vacuum':
            return np.array([avg_distance_norm, symmetry_balance_norm])

        # For fill mode, add vehicle type fractions
        total_positions = len(positions_with_labels)
        count_P = sum(1 for pos in positions_with_labels if pos[2].endswith('P'))
        count_B = sum(1 for pos in positions_with_labels if pos[2].endswith('B'))
        count_G = sum(1 for pos in positions_with_labels if pos[2].endswith('G'))

        fraction_P = count_P / total_positions if total_positions > 0 else 0.0
        fraction_B = count_B / total_positions if total_positions > 0 else 0.0
        fraction_G = count_G / total_positions if total_positions > 0 else 0.0

        return np.array([avg_distance_norm, symmetry_balance_norm,
                        fraction_P, fraction_B, fraction_G])

    @staticmethod
    def _compute_local_features(lattice: np.ndarray, i: int, j: int, label: str = None,
                                irradiation_mode: str = 'vacuum') -> List[float]:
        """Compute local features for a specific position using paper's definitions.

        Parameters
        ----------
        lattice : np.ndarray
            2D array representing the reactor core layout
        i : int
            Row position
        j : int
            Column position
        label : str, optional
            Position label (e.g., 'I_1P', 'I_2B') for fill mode
        irradiation_mode : str
            'vacuum' or 'fill'

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
        edge_dist = min(i, j, 7-i, 7-j)
        edge_dist_norm = edge_dist / 3.5  # Max is 3.5

        # 4. Distance to core center
        center_dist = np.sqrt((pos_center[0] - reactor_center[0])**2 +
                            (pos_center[1] - reactor_center[1])**2)
        max_dist = np.sqrt(2) * 4
        center_dist_norm = center_dist / max_dist

        # For vacuum mode, return 4 features
        if irradiation_mode == 'vacuum':
            return [local_fuel_density, coolant_contact_norm, edge_dist_norm, center_dist_norm]

        # For fill mode, add vehicle type as 5th feature
        vehicle_type = 0.0  # Default to P
        if label:
            if label.endswith('P'):
                vehicle_type = 0.0
            elif label.endswith('B'):
                vehicle_type = 0.5
            elif label.endswith('G'):
                vehicle_type = 1.0

        return [local_fuel_density, coolant_contact_norm, edge_dist_norm, center_dist_norm, vehicle_type]


    @staticmethod
    def _compute_nci_for_positions(positions: List[Tuple], lambda_decay: float = 1.5) -> List[float]:
        """Compute Neutron Competition Index for each irradiation position.

        NCI_i = sum over j≠i of contribution based on distance thresholds:
        - d < sqrt(4.9): exp(-(d_ij - 1) / lambda) (exponential decay, normalized to 1 at distance=1)
        - sqrt(4.9) <= d <= sqrt(5.1): exp(-(sqrt(5) - 1) / lambda) (lambda-dependent cutoff)
        - d > sqrt(5.1): 0 (no contribution)

        Note: The (distance - 1) normalization means adjacent cells (distance=1) have
        maximum interaction strength of exp(0) = 1.0

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

        # Distance thresholds (only used if cutoff enabled)
        threshold_low = np.sqrt(4.9)   # ~2.21
        threshold_high = np.sqrt(5.1)  # ~2.26

        # Medium distance contribution (evaluated at sqrt(5))
        medium_dist_contribution = np.exp(-(np.sqrt(5) - 1) / lambda_decay)

        nci_values = []
        for i, pos_i in enumerate(continuous_positions):
            nci = 0.0
            for j, pos_j in enumerate(continuous_positions):
                if i != j:
                    # Euclidean distance
                    dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)

                    # Formula is ALWAYS: exp(-(distance - 1) / lambda)
                    if NCI_DISTANCE_CUTOFF == 1:
                        # Apply knight's move cutoff
                        if dist < threshold_low:
                            nci += np.exp(-(dist - 1) / lambda_decay)
                        elif threshold_low <= dist <= threshold_high:
                            nci += medium_dist_contribution
                        # else: dist > threshold_high, contributes 0
                    elif NCI_DISTANCE_CUTOFF == 0:
                        # No cutoff: all distances contribute
                        nci += np.exp(-(dist - 1) / lambda_decay)
                    # NCI_DISTANCE_CUTOFF == 2: NCI not computed at all (handled in main function)

            nci_values.append(nci)

        return nci_values

    @staticmethod
    def _compute_nci_separate(positions_with_labels: List[Tuple],
                             lambda_P: float = 1.5,
                             lambda_B: float = 1.5,
                             lambda_G: float = 1.5) -> List[float]:
        """Compute separate NCI values by vehicle type (P, B, G) for each irradiation position.

        For each position, computes three NCI values with SEPARATE lambda values for each vehicle type:
        - NCI_P: competition from P-type vehicles (using lambda_P)
        - NCI_B: competition from B-type vehicles (using lambda_B)
        - NCI_G: competition from G-type vehicles (using lambda_G)

        Formula: NCI = exp(-(distance - 1) / lambda)
        Note: The (distance - 1) normalization means adjacent cells (distance=1) have
        maximum interaction strength of exp(0) = 1.0

        Parameters
        ----------
        positions_with_labels : List[Tuple]
            List of (row, col, label) tuples for irradiation positions
        lambda_P : float, optional
            Decay parameter for P-type vehicles (PWR), by default 1.5
        lambda_B : float, optional
            Decay parameter for B-type vehicles (BWR), by default 1.5
        lambda_G : float, optional
            Decay parameter for G-type vehicles (GAS), by default 1.5

        Returns
        -------
        List[float]
            List of NCI values: [NCI_P_pos1, NCI_B_pos1, NCI_G_pos1, NCI_P_pos2, ...]
        """
        # Convert to continuous coordinates (center of cells)
        continuous_positions = [(i + 0.5, j + 0.5, label) for i, j, label in positions_with_labels]

        # Distance thresholds (only used if cutoff enabled)
        threshold_low = np.sqrt(4.9)   # ~2.21
        threshold_high = np.sqrt(5.1)  # ~2.26

        # Medium distance contributions (evaluated at sqrt(5))
        medium_dist_contribution_P = np.exp(-(np.sqrt(5) - 1) / lambda_P)
        medium_dist_contribution_B = np.exp(-(np.sqrt(5) - 1) / lambda_B)
        medium_dist_contribution_G = np.exp(-(np.sqrt(5) - 1) / lambda_G)

        nci_values = []
        for i, pos_i in enumerate(continuous_positions):
            # Three separate NCI values for this position
            nci_P = 0.0
            nci_B = 0.0
            nci_G = 0.0

            for j, pos_j in enumerate(continuous_positions):
                if i != j:
                    # Euclidean distance
                    dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)

                    # Add to appropriate NCI based on vehicle type, using type-specific lambda
                    label_j = pos_j[2]

                    # Formula is ALWAYS: exp(-(distance - 1) / lambda)
                    if label_j.endswith('P'):
                        # Use lambda_P for P-type vehicles
                        if NCI_DISTANCE_CUTOFF == 1:
                            # Apply knight's move cutoff
                            if dist < threshold_low:
                                nci_P += np.exp(-(dist - 1) / lambda_P)
                            elif threshold_low <= dist <= threshold_high:
                                nci_P += medium_dist_contribution_P
                            # else: d > sqrt(5.1), contributes 0
                        elif NCI_DISTANCE_CUTOFF == 0:
                            # No cutoff: all distances contribute
                            nci_P += np.exp(-(dist - 1) / lambda_P)
                        # NCI_DISTANCE_CUTOFF == 2: NCI not computed (handled in main function)

                    elif label_j.endswith('B'):
                        # Use lambda_B for B-type vehicles
                        if NCI_DISTANCE_CUTOFF == 1:
                            # Apply knight's move cutoff
                            if dist < threshold_low:
                                nci_B += np.exp(-(dist - 1) / lambda_B)
                            elif threshold_low <= dist <= threshold_high:
                                nci_B += medium_dist_contribution_B
                            # else: d > sqrt(5.1), contributes 0
                        elif NCI_DISTANCE_CUTOFF == 0:
                            # No cutoff: all distances contribute
                            nci_B += np.exp(-(dist - 1) / lambda_B)
                        # NCI_DISTANCE_CUTOFF == 2: NCI not computed (handled in main function)

                    elif label_j.endswith('G'):
                        # Use lambda_G for G-type vehicles
                        if NCI_DISTANCE_CUTOFF == 1:
                            # Apply knight's move cutoff
                            if dist < threshold_low:
                                nci_G += np.exp(-(dist - 1) / lambda_G)
                            elif threshold_low <= dist <= threshold_high:
                                nci_G += medium_dist_contribution_G
                            # else: d > sqrt(5.1), contributes 0
                        elif NCI_DISTANCE_CUTOFF == 0:
                            # No cutoff: all distances contribute
                            nci_G += np.exp(-(dist - 1) / lambda_G)
                        # NCI_DISTANCE_CUTOFF == 2: NCI not computed (handled in main function)

            # Append all three NCI values for this position
            nci_values.extend([nci_P, nci_B, nci_G])

        return nci_values

    @staticmethod
    def spatial_convolution_encoding(lattice: np.ndarray, irradiation_mode: str = 'vacuum') -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
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
    def graph_based_encoding(lattice: np.ndarray, irradiation_mode: str = 'vacuum') -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
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
