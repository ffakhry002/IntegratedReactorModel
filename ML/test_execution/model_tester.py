"""
Model testing functionality - Fixed to handle position ordering and training data comparison
"""

import sys
import os
import numpy as np
import joblib
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.txt_to_data import parse_reactor_data
from ML_models.encodings.encoding_methods import ReactorEncodings


class ReactorModelTester:
    """Class to handle testing of reactor ML models"""

    def __init__(self, models_dir=None):
        self.encodings = ReactorEncodings()
        self.available_models = []
        self.test_results = []
        self.training_lattices = []

        # Set models directory
        if models_dir is None:
            self.models_dir = 'models'
        else:
            self.models_dir = models_dir

    def load_training_data(self, training_file_path="ML/data/train.txt"):
        """Load training data to check if test configurations were seen during training"""
        try:
            print(f"\nLoading training data from {training_file_path}...")
            # Parse training data
            parse_result = parse_reactor_data(training_file_path)
            if len(parse_result) == 4:
                # New format with descriptions
                lattices, _, _, _ = parse_result
            else:
                # Old format without descriptions
                lattices, _, _ = parse_result

            self.training_lattices = lattices
            print(f"Loaded {len(self.training_lattices)} training configurations")
            return True
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
            self.training_lattices = []
            return False

    def normalize_lattice_for_comparison(self, lattice):
        """
        Normalize a lattice for geometric comparison by replacing all irradiation
        labels with a generic 'I' marker
        """
        normalized = lattice.copy()
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                if normalized[i, j].startswith('I_'):
                    normalized[i, j] = 'I'  # Generic irradiation marker
        return normalized

    def lattices_match(self, lattice1, lattice2):
        """Check if two lattices are identical"""
        if lattice1.shape != lattice2.shape:
            return False

        # Compare element by element
        for i in range(lattice1.shape[0]):
            for j in range(lattice1.shape[1]):
                if lattice1[i, j] != lattice2[i, j]:
                    return False
        return True

    def lattices_match_geometrically(self, lattice1, lattice2):
        """
        Check if two lattices have the same geometric configuration,
        ignoring specific irradiation labels (I_1, I_2, etc.)
        """
        if lattice1.shape != lattice2.shape:
            return False

        # Normalize both lattices
        norm1 = self.normalize_lattice_for_comparison(lattice1)
        norm2 = self.normalize_lattice_for_comparison(lattice2)

        return self.lattices_match(norm1, norm2)

    def get_all_symmetries(self, lattice):
        """
        Generate all 8 symmetries of a lattice
        Returns list of (transformed_lattice, transformation_name) tuples
        """
        symmetries = []

        # Original
        symmetries.append((lattice.copy(), "original"))

        # 90 degree rotation
        symmetries.append((np.rot90(lattice, k=1), "rot90"))

        # 180 degree rotation
        symmetries.append((np.rot90(lattice, k=2), "rot180"))

        # 270 degree rotation
        symmetries.append((np.rot90(lattice, k=3), "rot270"))

        # Horizontal flip
        symmetries.append((np.fliplr(lattice), "flip_h"))

        # Vertical flip
        symmetries.append((np.flipud(lattice), "flip_v"))

        # Diagonal flip (transpose)
        symmetries.append((lattice.T, "transpose"))

        # Anti-diagonal flip
        symmetries.append((np.fliplr(lattice.T), "anti_diag"))

        return symmetries

    def is_in_training_set(self, test_lattice):
        """
        Check if a test lattice exists in the training set,
        considering all symmetries and ignoring irradiation label differences
        """
        # Get all symmetries of the test lattice
        test_symmetries = self.get_all_symmetries(test_lattice)

        # Check each training lattice
        for train_lattice in self.training_lattices:
            # Check if any symmetry of the test lattice matches the training lattice geometrically
            for test_sym, _ in test_symmetries:
                if self.lattices_match_geometrically(test_sym, train_lattice):
                    return True

        return False

    def find_matching_training_config(self, test_lattice):
        """
        Find which training configuration matches the test lattice (if any).
        Returns: (match_found, training_idx, transformation) or (False, None, None)
        """
        # Get all symmetries of the test lattice
        test_symmetries = self.get_all_symmetries(test_lattice)

        # Check each training lattice
        for train_idx, train_lattice in enumerate(self.training_lattices):
            # Check if any symmetry of the test lattice matches the training lattice geometrically
            for test_sym, transform_name in test_symmetries:
                if self.lattices_match_geometrically(test_sym, train_lattice):
                    return True, train_idx, transform_name

        return False, None, None

    def find_available_models(self):
        """Find all trained models in the models directory"""
        model_files = glob.glob(os.path.join(self.models_dir, '*.pkl'))

        for filepath in model_files:
            try:
                # Load model metadata
                model_data = joblib.load(filepath)

                # Extract metadata
                if 'model_class' in model_data:
                    # New format with full metadata
                    model_info = {
                        'filepath': filepath,
                        'model_class': model_data['model_class'],
                        'model_type': model_data['model_type'],  # flux or keff
                        'encoding': model_data['encoding'],
                        'optimization_method': model_data.get('optimization_method', 'unknown'),
                        'model': model_data.get('model'),
                        'use_log_flux': model_data.get('use_log_flux', False),
                        'flux_scale': model_data.get('flux_scale', 1e14)
                    }
                else:
                    # Try to parse from filename for backward compatibility
                    filename = os.path.basename(filepath)
                    parts = filename.replace('.pkl', '').split('_')

                    model_info = {
                        'filepath': filepath,
                        'model_class': parts[0] if len(parts) > 0 else 'unknown',
                        'model_type': parts[1] if len(parts) > 1 else 'unknown',
                        'encoding': parts[2] if len(parts) > 2 else 'unknown',
                        'optimization_method': parts[3] if len(parts) > 3 else 'unknown',
                        'model': model_data.get('model', model_data),
                        'use_log_flux': False,  # Default for old models
                        'flux_scale': 1e14
                    }

                self.available_models.append(model_info)

            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")

        return self.available_models

    def encode_lattice(self, lattice, encoding_method):
        """Encode a lattice using the specified method - returns position order"""
        if encoding_method == 'one_hot':
            return self.encodings.one_hot_encoding(lattice)
        elif encoding_method == 'categorical':
            return self.encodings.categorical_encoding(lattice)
        elif encoding_method == 'physics':
            return self.encodings.physics_based_encoding(lattice)
        elif encoding_method == 'spatial':
            return self.encodings.spatial_convolution_encoding(lattice)
        elif encoding_method == 'graph':
            return self.encodings.graph_based_encoding(lattice)
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")

    def test_single_configuration(self, lattice, flux_actual, keff_actual, config_id, description="", in_training_set=False):
        """Test all available models on a single configuration"""
        results = []

        for model_info in self.available_models:
            try:
                # Encode lattice - now returns position_order too
                encode_result = self.encode_lattice(lattice, model_info['encoding'])

                # Handle both old and new encoding return formats
                if len(encode_result) == 3:
                    features, irr_positions, position_order = encode_result
                else:
                    # Old format - create position order from irr_positions
                    features, irr_positions = encode_result
                    position_order = sorted(irr_positions)

                features = features.reshape(1, -1)

                # Load model data
                model_data = joblib.load(model_info['filepath'])
                model = model_data.get('model')

                result = {
                    'in_training': 'T' if in_training_set else 'F',  # Add this as first field
                    'config_id': config_id,
                    'description': description,
                    'model_class': model_info['model_class'],
                    'model_type': model_info['model_type'],
                    'encoding': model_info['encoding'],
                    'optimization_method': model_info['optimization_method']
                }

                if model_info['model_type'] == 'flux':
                    # Check if model needs scaling
                    if 'scaler' in model_data:
                        features = model_data['scaler'].transform(features)

                    # Predict flux
                    flux_pred = model.predict(features)
                    if len(flux_pred.shape) == 1:
                        flux_pred = flux_pred.reshape(1, -1)
                    flux_pred = flux_pred[0]  # Get first (and only) sample

                    # Check if model was trained with log transform
                    use_log_flux = model_info.get('use_log_flux', model_data.get('use_log_flux', False))
                    flux_scale = model_info.get('flux_scale', model_data.get('flux_scale', 1e14))

                    # Transform predictions back to original scale
                    if use_log_flux:
                        # Convert from log scale to original scale
                        flux_pred_original = 10 ** flux_pred
                    else:
                        # Scale back if needed
                        flux_pred_original = flux_pred * flux_scale

                    # Map predictions to positions
                    position_to_flux_pred = {}
                    for idx, pos in enumerate(position_order):
                        if idx < len(flux_pred_original):
                            position_to_flux_pred[pos] = flux_pred_original[idx]

                    # Now map positions back to labels in the test lattice
                    label_to_flux_pred = {}
                    for i in range(lattice.shape[0]):
                        for j in range(lattice.shape[1]):
                            if lattice[i, j].startswith('I') and (i, j) in position_to_flux_pred:
                                label_to_flux_pred[lattice[i, j]] = position_to_flux_pred[(i, j)]

                    # Store flux results for each irradiation position
                    flux_pred_list = []
                    flux_actual_list = []

                    for i in range(1, 5):
                        label = f'I_{i}'
                        if label in label_to_flux_pred and label in flux_actual:
                            pred_val = label_to_flux_pred[label]
                            actual_val = flux_actual[label]

                            result[f'I_{i}_real'] = actual_val
                            result[f'I_{i}_predicted'] = pred_val
                            result[f'I_{i}_rel_error'] = abs((pred_val - actual_val) / actual_val * 100) if actual_val != 0 else 0

                            flux_pred_list.append(pred_val)
                            flux_actual_list.append(actual_val)

                    # Calculate average flux
                    if flux_pred_list and flux_actual_list:
                        avg_pred = np.mean(flux_pred_list)
                        avg_actual = np.mean(flux_actual_list)

                        result['avg_flux_real'] = avg_actual
                        result['avg_flux_predicted'] = avg_pred
                        result['avg_flux_rel_error'] = abs((avg_pred - avg_actual) / avg_actual * 100) if avg_actual != 0 else 0

                        # Calculate MAPE
                        result['mape_flux'] = np.mean([abs((p - a) / a) * 100 for p, a in zip(flux_pred_list, flux_actual_list) if a != 0])

                elif model_info['model_type'] == 'keff':
                    # Check if model needs scaling
                    if 'scaler' in model_data:
                        features = model_data['scaler'].transform(features)

                    # Predict k-eff
                    keff_pred = model.predict(features)
                    if hasattr(keff_pred, '__len__'):
                        keff_pred = keff_pred[0]

                    result['keff_real'] = keff_actual
                    result['keff_predicted'] = keff_pred
                    result['keff_rel_error'] = abs((keff_pred - keff_actual) / keff_actual * 100) if keff_actual != 0 else 0

                    # Store MAPE for internal use
                    result['mape_keff'] = result['keff_rel_error']

                results.append(result)

            except Exception as e:
                print(f"\nError testing {model_info['model_class']} {model_info['model_type']}: {e}")
                import traceback
                traceback.print_exc()

        return results

    def test_file(self, test_file_path, training_file_path="ML/data/train.txt", show_match_details=False):
        """Test all models on a test file"""
        print(f"\nLoading test data from {test_file_path}...")

        # Load training data for comparison
        self.load_training_data(training_file_path)

        # Parse test data - try to get descriptions if available
        try:
            # Try parsing with descriptions
            parse_result = parse_reactor_data(test_file_path)
            if len(parse_result) == 4:
                # New format with descriptions
                lattices, flux_data, k_effectives, descriptions = parse_result
            else:
                # Old format without descriptions
                lattices, flux_data, k_effectives = parse_result
                descriptions = [""] * len(lattices)  # Empty descriptions
        except:
            # Fallback to old format
            lattices, flux_data, k_effectives = parse_reactor_data(test_file_path)
            descriptions = [""] * len(lattices)  # Empty descriptions

        print(f"Found {len(lattices)} test configurations")

        # Test each configuration
        all_results = []
        match_count = 0

        for i, (lattice, flux, keff, desc) in enumerate(zip(lattices, flux_data, k_effectives, descriptions)):
            print(f"\rTesting configuration {i+1}/{len(lattices)}...", end='')

            # Check if this configuration was in training set
            in_training = self.is_in_training_set(lattice)

            if in_training:
                match_count += 1

            # Optionally show detailed match information
            if show_match_details and in_training:
                match_found, train_idx, transform = self.find_matching_training_config(lattice)
                print(f"\n  Config {i+1} matches training config {train_idx+1} with transform: {transform}")

            results = self.test_single_configuration(lattice, flux, keff, i, desc, in_training)
            all_results.extend(results)

        print(f"\nTesting complete!")
        print(f"\nSummary: {match_count}/{len(lattices)} test configurations were seen during training (considering symmetries)")

        return all_results
