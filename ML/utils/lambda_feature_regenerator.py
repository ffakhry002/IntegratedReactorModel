"""
Feature regeneration utilities for lambda-aware NCI optimization.

This module provides efficient regeneration of NCI features during hyperparameter
optimization without recomputing all features or causing CV data leakage.
"""

import numpy as np
from typing import Dict, Tuple, List
from ML_models.encodings.encoding_methods import ReactorEncodings


class LambdaFeatureRegenerator:
    """Efficiently regenerate NCI features with different lambda values.

    Key principles:
    1. Only regenerate lambda-dependent features (NCI values)
    2. Cache lambda-independent features (global and local features)
    3. Ensure CV integrity by only using training fold data
    4. Minimize redundant computation
    """

    def __init__(self, encoding_method: str):
        """Initialize the feature regenerator.

        Parameters
        ----------
        encoding_method : str
            Encoding method being used ('physics', 'one_hot', etc.)
        """
        self.encoding_method = encoding_method
        self.cache = {}
        self.supports_lambda = encoding_method == 'physics'

    def extract_nci_indices(self, total_features: int, irradiation_mode: str,
                           nci_mode: str, n_positions: int = 4) -> Tuple[int, int]:
        """Determine which feature indices correspond to NCI values.

        For physics encoding:
        - Vacuum mode: 2 global + 4*4 local + 4 NCI = 22 features
          NCI indices: [18:22]
        - Fill mode (single NCI): 5 global + 4*5 local + 4 NCI = 29 features
          NCI indices: [25:29]
        - Fill mode (separate NCI): 5 global + 4*5 local + 4*3 NCI = 37 features
          NCI indices: [25:37]

        Parameters
        ----------
        total_features : int
            Total number of features
        irradiation_mode : str
            'vacuum' or 'fill'
        nci_mode : str
            'single' or 'separate'
        n_positions : int
            Number of irradiation positions (default: 4)

        Returns
        -------
        Tuple[int, int]
            (start_index, end_index) for NCI features
        """
        if irradiation_mode == 'vacuum':
            # 2 global + 4*4 local = 18 non-NCI features
            nci_start = 2 + 4 * n_positions
            nci_end = nci_start + n_positions  # 4 NCI values
        elif nci_mode == 'separate':
            # 5 global + 5*4 local = 25 non-NCI features
            nci_start = 5 + 5 * n_positions
            nci_end = nci_start + 3 * n_positions  # 12 NCI values (3 per position)
        else:  # fill mode, single NCI
            # 5 global + 5*4 local = 25 non-NCI features
            nci_start = 5 + 5 * n_positions
            nci_end = nci_start + n_positions  # 4 NCI values

        return nci_start, nci_end

    def separate_features(self, X: np.ndarray, lattices: List[np.ndarray],
                         irradiation_mode: str, nci_mode: str) -> Dict:
        """Separate features into lambda-independent and lambda-dependent parts.

        Parameters
        ----------
        X : np.ndarray
            Full feature matrix (n_samples, n_features)
        lattices : List[np.ndarray]
            List of lattice configurations (for NCI regeneration)
        irradiation_mode : str
            'vacuum' or 'fill'
        nci_mode : str
            'single' or 'separate'

        Returns
        -------
        Dict
            Dictionary containing:
            - 'X_fixed': Features that don't depend on lambda
            - 'nci_start': Start index for NCI features
            - 'nci_end': End index for NCI features
            - 'lattices': Lattice configurations for regeneration
        """
        if not self.supports_lambda:
            return {
                'X_fixed': X,
                'nci_start': None,
                'nci_end': None,
                'lattices': None
            }

        n_features = X.shape[1]
        nci_start, nci_end = self.extract_nci_indices(
            n_features, irradiation_mode, nci_mode)

        # Extract non-NCI features (these are lambda-independent)
        X_before_nci = X[:, :nci_start]
        X_after_nci = X[:, nci_end:] if nci_end < n_features else np.array([]).reshape(X.shape[0], 0)

        X_fixed = np.hstack([X_before_nci, X_after_nci]) if X_after_nci.shape[1] > 0 else X_before_nci

        return {
            'X_fixed': X_fixed,
            'nci_start': nci_start,
            'nci_end': nci_end,
            'lattices': lattices,
            'irradiation_mode': irradiation_mode,
            'nci_mode': nci_mode
        }

    def regenerate_features(self, feature_data: Dict, lambda_decay: float = 1.5,
                           lambda_P: float = 1.5, lambda_B: float = 1.5,
                           lambda_G: float = 1.5) -> np.ndarray:
        """Regenerate features with new lambda values.

        Parameters
        ----------
        feature_data : Dict
            Dictionary from separate_features() containing fixed features and lattices
        lambda_decay : float
            Lambda for single NCI mode
        lambda_P : float
            Lambda for P-type vehicles
        lambda_B : float
            Lambda for B-type vehicles
        lambda_G : float
            Lambda for G-type vehicles

        Returns
        -------
        np.ndarray
            Reconstructed feature matrix with new NCI values
        """
        if not self.supports_lambda:
            return feature_data['X_fixed']

        X_fixed = feature_data['X_fixed']
        lattices = feature_data['lattices']
        nci_start = feature_data['nci_start']
        irradiation_mode = feature_data['irradiation_mode']
        nci_mode = feature_data['nci_mode']

        # Generate new NCI features for each lattice
        new_nci_features = []

        for lattice in lattices:
            # Extract positions
            irr_positions = []
            irr_positions_with_labels = []

            for i in range(lattice.shape[0]):
                for j in range(lattice.shape[1]):
                    if lattice[i, j].startswith('I'):
                        irr_positions.append((i, j))
                        irr_positions_with_labels.append((i, j, lattice[i, j]))

            position_order = sorted(irr_positions)
            position_order_with_labels = sorted(irr_positions_with_labels,
                                               key=lambda x: (x[0], x[1]))

            # Compute NCI features with new lambda values
            if irradiation_mode == 'fill' and nci_mode == 'separate':
                nci_values = ReactorEncodings._compute_nci_separate(
                    position_order_with_labels, lambda_P, lambda_B, lambda_G)
            else:
                nci_values = ReactorEncodings._compute_nci_for_positions(
                    position_order, lambda_decay)

            new_nci_features.append(nci_values)

        X_nci = np.array(new_nci_features)

        # Reconstruct full feature matrix: [X_fixed_before | X_nci | X_fixed_after]
        # Note: We extracted X_fixed as [before_nci, after_nci] concatenated
        # So we need to insert NCI in the middle

        # Split X_fixed back into before and after parts
        n_samples = X_fixed.shape[0]
        n_features_before = nci_start
        n_features_after = X_fixed.shape[1] - n_features_before

        X_before = X_fixed[:, :n_features_before]
        X_after = X_fixed[:, n_features_before:] if n_features_after > 0 else np.array([]).reshape(n_samples, 0)

        # Reconstruct
        if X_after.shape[1] > 0:
            X_reconstructed = np.hstack([X_before, X_nci, X_after])
        else:
            X_reconstructed = np.hstack([X_before, X_nci])

        return X_reconstructed

    def create_cv_safe_regenerator(self, X_train_full: np.ndarray,
                                   lattices_full: List[np.ndarray],
                                   train_indices: np.ndarray,
                                   irradiation_mode: str,
                                   nci_mode: str) -> Tuple[Dict, np.ndarray]:
        """Create a CV-safe feature regenerator for a specific fold.

        This ensures no data leakage by only using training fold data.

        Parameters
        ----------
        X_train_full : np.ndarray
            Full training data (before CV split)
        lattices_full : List[np.ndarray]
            Full list of lattices (before CV split)
        train_indices : np.ndarray
            Indices for this CV fold's training set
        irradiation_mode : str
            'vacuum' or 'fill'
        nci_mode : str
            'single' or 'separate'

        Returns
        -------
        Tuple[Dict, np.ndarray]
            (feature_data_for_fold, fold_lattices)
        """
        # Extract only the training fold data
        X_fold = X_train_full[train_indices]
        lattices_fold = [lattices_full[i] for i in train_indices]

        # Separate features for this fold only
        feature_data = self.separate_features(
            X_fold, lattices_fold, irradiation_mode, nci_mode)

        return feature_data, lattices_fold


def get_lambda_params_for_encoding(irradiation_mode: str, nci_mode: str) -> List[str]:
    """Get list of lambda parameter names for a given encoding configuration.

    Parameters
    ----------
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    List[str]
        List of lambda parameter names to optimize
    """
    if irradiation_mode == 'vacuum':
        # Vacuum mode uses single lambda
        return ['lambda_decay']
    elif nci_mode == 'separate':
        # Fill mode with separate NCI uses three lambdas
        return ['lambda_P', 'lambda_B', 'lambda_G']
    else:
        # Fill mode with single NCI uses one lambda
        return ['lambda_decay']


def extract_lambda_from_trial_params(trial_params: Dict, irradiation_mode: str,
                                     nci_mode: str) -> Dict:
    """Extract lambda values from trial parameters.

    Parameters
    ----------
    trial_params : Dict
        Trial parameters from Optuna or other optimizer
    irradiation_mode : str
        'vacuum' or 'fill'
    nci_mode : str
        'single' or 'separate'

    Returns
    -------
    Dict
        Dictionary with lambda parameter names as keys
    """
    param_names = get_lambda_params_for_encoding(irradiation_mode, nci_mode)
    return {name: trial_params.get(name, 1.5) for name in param_names}
