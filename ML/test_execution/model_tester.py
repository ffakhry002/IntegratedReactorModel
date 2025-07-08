"""
Model testing functionality - Fixed to handle position ordering and training data comparison
UPDATED: Ensures label-agnostic testing with proper label mapping for output
NEW: Supports different flux modes (total, energy, bin)
"""

import sys
import os
import numpy as np
import joblib
import glob
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.txt_to_data import parse_reactor_data
from ML_models.encodings.encoding_methods import ReactorEncodings


class ReactorModelTester:
    """Class to handle testing of reactor ML models"""

    def __init__(self, models_dir=None):
        """Initialize the reactor model tester.

        Parameters
        ----------
        models_dir : str, optional
            Directory containing trained models. If None, uses default 'models'.

        Returns
        -------
        None
        """
        self.encodings = ReactorEncodings()
        self.available_models = []
        self.test_results = []
        self.training_lattices = []
        self.total_flux_results = {}  # Cache for bin mode pairing

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
            if len(parse_result) == 5:
                # New format with energy groups
                lattices, _, _, _, _ = parse_result
            elif len(parse_result) == 4:
                # Format with descriptions
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
                # Parse model info from filename first (most efficient)
                filename = os.path.basename(filepath)
                parts = filename.replace('.pkl', '').split('_')

                model_info = self._parse_model_info_from_filename(parts, filepath)

                # Only load the actual model file if we need metadata that's not in filename
                # For now, we'll use filename parsing and set reasonable defaults
                # The actual model loading will happen later when the model is selected for use

                self.available_models.append(model_info)

            except Exception as e:
                print(f"Warning: Could not process {filepath}: {e}")

        return self.available_models

    def _parse_model_info_from_filename(self, parts, filepath):
        """Parse model information from filename without loading the actual model"""
        # Check if flux mode is in filename
        flux_mode = 'total'
        if len(parts) >= 3 and parts[1] == 'flux':
            # Check for new single energy modes
            if parts[2] == 'thermal' and len(parts) > 3 and parts[3] == 'only':
                flux_mode = 'thermal_only'
                # Adjust parsing for model_flux_thermal_only_encoding_optimization.pkl
                model_info = {
                    'filepath': filepath,
                    'model_class': parts[0],
                    'model_type': 'flux',
                    'flux_mode': flux_mode,
                    'encoding': parts[4] if len(parts) > 4 else 'unknown',
                    'optimization_method': parts[5] if len(parts) > 5 else 'unknown',
                    'use_log_flux': True,  # Default for single energy modes
                    'flux_scale': 1e14
                }
            elif parts[2] == 'epithermal' and len(parts) > 3 and parts[3] == 'only':
                flux_mode = 'epithermal_only'
                model_info = {
                    'filepath': filepath,
                    'model_class': parts[0],
                    'model_type': 'flux',
                    'flux_mode': flux_mode,
                    'encoding': parts[4] if len(parts) > 4 else 'unknown',
                    'optimization_method': parts[5] if len(parts) > 5 else 'unknown',
                    'use_log_flux': True,  # Default for single energy modes
                    'flux_scale': 1e14
                }
            elif parts[2] == 'fast' and len(parts) > 3 and parts[3] == 'only':
                flux_mode = 'fast_only'
                model_info = {
                    'filepath': filepath,
                    'model_class': parts[0],
                    'model_type': 'flux',
                    'flux_mode': flux_mode,
                    'encoding': parts[4] if len(parts) > 4 else 'unknown',
                    'optimization_method': parts[5] if len(parts) > 5 else 'unknown',
                    'use_log_flux': True,  # Default for single energy modes
                    'flux_scale': 1e14
                }
            elif parts[2] in ['energy', 'bin', 'total']:
                flux_mode = parts[2]
                model_info = {
                    'filepath': filepath,
                    'model_class': parts[0],
                    'model_type': 'flux',
                    'flux_mode': flux_mode,
                    'encoding': parts[3] if len(parts) > 3 else 'unknown',
                    'optimization_method': parts[4] if len(parts) > 4 else 'unknown',
                    'use_log_flux': True,  # Default for multi-energy modes
                    'flux_scale': 1e14
                }
            else:
                # Default parsing for backward compatibility
                model_info = {
                    'filepath': filepath,
                    'model_class': parts[0] if len(parts) > 0 else 'unknown',
                    'model_type': parts[1] if len(parts) > 1 else 'unknown',
                    'encoding': parts[2] if len(parts) > 2 else 'unknown',
                    'optimization_method': parts[3] if len(parts) > 3 else 'unknown',
                    'use_log_flux': False,
                    'flux_scale': 1e14,
                    'flux_mode': 'total'
                }
        else:
            # Standard format: model_type_encoding_optimization.pkl
            model_info = {
                'filepath': filepath,
                'model_class': parts[0] if len(parts) > 0 else 'unknown',
                'model_type': parts[1] if len(parts) > 1 else 'unknown',
                'encoding': parts[2] if len(parts) > 2 else 'unknown',
                'optimization_method': parts[3] if len(parts) > 3 else 'unknown',
                'use_log_flux': False,
                'flux_scale': 1e14,
                'flux_mode': 'total'
            }

        return model_info

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

    def load_total_flux_results(self, excel_path):
        """Load total flux results from an Excel file for bin mode pairing"""
        try:
            df = pd.read_excel(excel_path, sheet_name='Test Results')

            # Create a dictionary keyed by (config_id, model_class, encoding, optimization_method)
            results_dict = {}
            for _, row in df.iterrows():
                key = (
                    row['config_id'],
                    row['model_class'],
                    row['encoding'],
                    row['optimization_method']
                )

                # Store the total flux predictions
                flux_data = {}
                for i in range(1, 5):
                    if f'I_{i}_predicted' in row:
                        flux_data[f'I_{i}'] = {
                            'predicted': row[f'I_{i}_predicted'],
                            'real': row[f'I_{i}_real']
                        }

                results_dict[key] = flux_data

            self.total_flux_results = results_dict
            print(f"Loaded {len(results_dict)} total flux results from {excel_path}")
            return True

        except Exception as e:
            print(f"Error loading total flux results: {e}")
            return False

    def test_single_configuration(self, lattice, flux_actual, keff_actual, config_id, description="",
                                in_training_set=False, energy_groups=None):
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

                # Load model data - now we load the actual model file
                model_data = joblib.load(model_info['filepath'])

                # Update model_info with actual metadata from file if available
                if 'model_class' in model_data:
                    # Update with actual metadata from file
                    model_info.update({
                        'model_class': model_data['model_class'],
                        'model_type': model_data['model_type'],
                        'encoding': model_data['encoding'],
                        'optimization_method': model_data.get('optimization_method', model_info.get('optimization_method', 'unknown')),
                        'use_log_flux': model_data.get('use_log_flux', model_info.get('use_log_flux', False)),
                        'flux_scale': model_data.get('flux_scale', model_info.get('flux_scale', 1e14)),
                        'flux_mode': model_data.get('flux_mode', model_info.get('flux_mode', 'total'))
                    })

                # CRITICAL FIX: Properly load model using the appropriate model class
                model_class_name = model_info.get('model_class', 'unknown')

                # Import the appropriate model class
                if model_class_name == 'xgboost':
                    from ML_models import XGBoostReactorModel
                    model_wrapper, metadata = XGBoostReactorModel.load_model(model_info['filepath'])
                elif model_class_name == 'random_forest':
                    from ML_models import RandomForestReactorModel
                    model_wrapper, metadata = RandomForestReactorModel.load_model(model_info['filepath'])
                elif model_class_name == 'svm':
                    from ML_models import SVMReactorModel
                    model_wrapper, metadata = SVMReactorModel.load_model(model_info['filepath'])
                elif model_class_name == 'neural_net':
                    from ML_models import NeuralNetReactorModel
                    model_wrapper, metadata = NeuralNetReactorModel.load_model(model_info['filepath'])
                else:
                    # Fallback to old method for backward compatibility
                    print(f"Warning: Unknown model class '{model_class_name}', using raw model")
                    model_wrapper = None
                    metadata = {}

                result = {
                    'in_training': 'T' if in_training_set else 'F',  # Add this as first field
                    'config_id': config_id,
                    'description': description,
                    'model_class': model_info['model_class'],
                    'model_type': model_info['model_type'],
                    'encoding': model_info['encoding'],
                    'optimization_method': model_info['optimization_method'],
                    'flux_mode': model_info.get('flux_mode', 'total')  # NEW: Add flux mode to results
                }

                if model_info['model_type'] == 'flux':
                    # Check if model needs scaling (for backward compatibility with raw models)
                    if model_wrapper is None:
                        # Old raw model approach
                        model = model_data.get('model')
                        if 'scaler' in model_data:
                            features = model_data['scaler'].transform(features)
                        flux_pred = model.predict(features)
                    else:
                        # New wrapper approach - use the wrapper's predict method
                        flux_pred = model_wrapper.predict_flux(features)

                    if len(flux_pred.shape) == 1:
                        flux_pred = flux_pred.reshape(1, -1)
                    flux_pred = flux_pred[0]  # Get first (and only) sample

                    # Check flux mode
                    flux_mode = model_info.get('flux_mode', 'total')

                    if flux_mode == 'total':
                        # Original handling for total flux
                        # Check if model was trained with log transform
                        use_log_flux = model_info.get('use_log_flux', model_data.get('use_log_flux', False))
                        flux_scale = model_info.get('flux_scale', model_data.get('flux_scale', 1e14))

                        # Transform predictions back to original scale
                        # Both raw models and wrappers predict in transformed space
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

                        # Map positions to labels
                        label_to_flux_pred = {}
                        for i in range(lattice.shape[0]):
                            for j in range(lattice.shape[1]):
                                if lattice[i, j].startswith('I_') and (i, j) in position_to_flux_pred:
                                    label = lattice[i, j]
                                    label_to_flux_pred[label] = position_to_flux_pred[(i, j)]

                        # Calculate errors for each position
                        flux_pred_list = []
                        flux_actual_list = []

                        # FIXED: Iterate through actual labels in the lattice instead of assuming I_1 through I_4
                        # This handles configurations where labels appear in non-standard positions
                        labels_found = []
                        for row in range(lattice.shape[0]):
                            for col in range(lattice.shape[1]):
                                if lattice[row, col].startswith('I_'):
                                    labels_found.append(lattice[row, col])

                        # Remove duplicates while preserving order
                        seen = set()
                        unique_labels = []
                        for label in labels_found:
                            if label not in seen:
                                seen.add(label)
                                unique_labels.append(label)

                        # Process each label found in the lattice
                        for label in sorted(unique_labels):  # Sort to ensure consistent output order
                            if label in label_to_flux_pred and label in flux_actual:
                                pred_val = label_to_flux_pred[label]
                                actual_val = flux_actual[label]

                                # Extract position number from label (e.g., 'I_3' -> 3)
                                label_num = int(label.split('_')[1])

                                result[f'I_{label_num}_real'] = actual_val
                                result[f'I_{label_num}_predicted'] = pred_val
                                # SIGNED MAPE: positive = overprediction, negative = underprediction
                                result[f'I_{label_num}_rel_error'] = ((pred_val - actual_val) / actual_val * 100) if actual_val != 0 else 0

                                flux_pred_list.append(pred_val)
                                flux_actual_list.append(actual_val)

                        # Calculate average flux
                        if flux_pred_list and flux_actual_list:
                            avg_pred = np.mean(flux_pred_list)
                            avg_actual = np.mean(flux_actual_list)

                            result['avg_flux_real'] = avg_actual
                            result['avg_flux_predicted'] = avg_pred
                            # SIGNED MAPE: positive = overprediction, negative = underprediction
                            result['avg_flux_rel_error'] = ((avg_pred - avg_actual) / avg_actual * 100) if avg_actual != 0 else 0
                            result['mape_flux'] = np.mean([((p - a) / a) * 100 for p, a in zip(flux_pred_list, flux_actual_list) if a != 0])

                    elif flux_mode in ['energy', 'bin']:
                        # Handle energy and bin modes (12 outputs)
                        # Transform predictions if needed
                        if flux_mode == 'energy':
                            use_log_flux = model_info.get('use_log_flux', model_data.get('use_log_flux', False))
                            flux_scale = model_info.get('flux_scale', model_data.get('flux_scale', 1e14))

                            if use_log_flux:
                                flux_pred = 10 ** flux_pred
                            else:
                                flux_pred = flux_pred * flux_scale

                        # For bin mode, predictions are already fractions (0-1)

                        # Map the 12 predictions to positions and energy groups
                        # Order: [pos1_thermal, pos1_epithermal, pos1_fast, pos2_thermal, ...]
                        energy_groups_model = ['thermal', 'epithermal', 'fast']

                        # Map predictions to position and energy group
                        position_energy_pred = {}
                        pred_idx = 0
                        for pos in position_order:
                            position_energy_pred[pos] = {}
                            for energy in energy_groups_model:
                                if pred_idx < len(flux_pred):
                                    position_energy_pred[pos][energy] = flux_pred[pred_idx]
                                    pred_idx += 1

                        # Map positions to labels and prepare results
                        for i in range(lattice.shape[0]):
                            for j in range(lattice.shape[1]):
                                if lattice[i, j].startswith('I_') and (i, j) in position_energy_pred:
                                    label = lattice[i, j]

                                    if flux_mode == 'energy':
                                        # For energy mode, use predicted absolute fluxes directly
                                        for energy in energy_groups_model:
                                            pred_val = position_energy_pred[(i, j)][energy]

                                            # Get actual values from energy_groups
                                            if energy_groups and label in energy_groups:
                                                actual_fraction = energy_groups[label][energy]
                                                actual_val = flux_actual[label] * actual_fraction
                                            else:
                                                actual_val = 0

                                            result[f'{label}_{energy}_real'] = actual_val
                                            result[f'{label}_{energy}_predicted'] = pred_val
                                            # SIGNED MAPE: positive = overprediction, negative = underprediction
                                            result[f'{label}_{energy}_rel_error'] = ((pred_val - actual_val) / actual_val * 100) if actual_val != 0 else 0

                                        # Calculate total flux
                                        total_pred = sum(position_energy_pred[(i, j)].values())
                                        total_real = flux_actual.get(label, 0)

                                        result[f'{label}_total_real'] = total_real
                                        result[f'{label}_total_predicted'] = total_pred
                                        # SIGNED MAPE: positive = overprediction, negative = underprediction
                                        result[f'{label}_total_rel_error'] = ((total_pred - total_real) / total_real * 100) if total_real != 0 else 0

                                    elif flux_mode == 'bin':
                                        # For bin mode, need to multiply by total flux
                                        # First check if we have total flux results loaded
                                        key = (config_id, model_info['model_class'], model_info['encoding'],
                                              model_info['optimization_method'])

                                        if key in self.total_flux_results and label in self.total_flux_results[key]:
                                            total_flux_pred = self.total_flux_results[key][label]['predicted']
                                            total_flux_real = self.total_flux_results[key][label]['real']

                                            for energy in energy_groups_model:
                                                # Predicted absolute flux = bin fraction * total flux
                                                bin_fraction = position_energy_pred[(i, j)][energy]
                                                pred_val = bin_fraction * total_flux_pred

                                                # Get actual values
                                                if energy_groups and label in energy_groups:
                                                    actual_fraction = energy_groups[label][energy]
                                                    actual_val = flux_actual[label] * actual_fraction
                                                else:
                                                    actual_val = 0

                                                result[f'{label}_{energy}_real'] = actual_val
                                                result[f'{label}_{energy}_predicted'] = pred_val
                                                # SIGNED MAPE: positive = overprediction, negative = underprediction
                                                result[f'{label}_{energy}_rel_error'] = ((pred_val - actual_val) / actual_val * 100) if actual_val != 0 else 0

                                            # For bin mode, total flux comes from the linked file
                                            result[f'{label}_total_real'] = total_flux_real
                                            result[f'{label}_total_predicted'] = total_flux_pred
                                            # SIGNED MAPE: positive = overprediction, negative = underprediction
                                            result[f'{label}_total_rel_error'] = ((total_flux_pred - total_flux_real) / total_flux_real * 100) if total_flux_real != 0 else 0
                                        else:
                                            # No total flux data available for this configuration
                                            print(f"\nWarning: No total flux data for config {config_id}, {model_info['model_class']}, {model_info['encoding']}, {model_info['optimization_method']}")
                                            for energy in energy_groups_model:
                                                result[f'{label}_{energy}_real'] = 'N/A'
                                                result[f'{label}_{energy}_predicted'] = 'N/A'
                                                result[f'{label}_{energy}_rel_error'] = 'N/A'
                                            result[f'{label}_total_real'] = 'N/A'
                                            result[f'{label}_total_predicted'] = 'N/A'
                                            result[f'{label}_total_rel_error'] = 'N/A'

                        # Calculate MAPE for energy/bin modes
                        if flux_mode in ['energy', 'bin']:
                            all_errors = []
                            # FIXED: Collect all errors from actual labels in results
                            for key in result:
                                if key.endswith('_rel_error') and not key.startswith('avg') and result[key] != 'N/A':
                                    # Check if it's an energy-specific error (contains thermal/epithermal/fast)
                                    if any(energy in key for energy in energy_groups_model):
                                        all_errors.append(result[key])

                            if all_errors:
                                result['mape_flux'] = np.mean(all_errors)
                            else:
                                result['mape_flux'] = 'N/A'

                    elif flux_mode in ['thermal_only', 'epithermal_only', 'fast_only']:
                        # NEW: Handle single energy group modes
                        # Transform predictions back from log scale if needed
                        use_log_flux = model_info.get('use_log_flux', model_data.get('use_log_flux', False))
                        flux_scale = model_info.get('flux_scale', model_data.get('flux_scale', 1e14))

                        if use_log_flux:
                            flux_pred_original = 10 ** flux_pred
                        else:
                            flux_pred_original = flux_pred * flux_scale

                        # Determine which energy group we're testing
                        energy_group = flux_mode.replace('_only', '')

                        # Map predictions to positions
                        position_to_flux_pred = {}
                        for idx, pos in enumerate(position_order):
                            if idx < len(flux_pred_original):
                                position_to_flux_pred[pos] = flux_pred_original[idx]

                        # Map positions to labels
                        label_to_flux_pred = {}
                        for i in range(lattice.shape[0]):
                            for j in range(lattice.shape[1]):
                                if lattice[i, j].startswith('I_') and (i, j) in position_to_flux_pred:
                                    label = lattice[i, j]
                                    label_to_flux_pred[label] = position_to_flux_pred[(i, j)]

                        # Calculate errors for each position
                        flux_pred_list = []
                        flux_actual_list = []
                        warned_no_energy_data = False  # Track if we've warned about missing energy data

                        # FIXED: Use actual labels from lattice instead of assuming I_1 through I_4
                        labels_found = []
                        for row in range(lattice.shape[0]):
                            for col in range(lattice.shape[1]):
                                if lattice[row, col].startswith('I_'):
                                    labels_found.append(lattice[row, col])

                        # Remove duplicates
                        unique_labels = list(dict.fromkeys(labels_found))

                        for label in sorted(unique_labels):
                            if label in label_to_flux_pred and label in flux_actual:
                                pred_val = label_to_flux_pred[label]

                                # Check if we have energy group data
                                if energy_groups and label in energy_groups and energy_group in energy_groups[label]:
                                    # Actual value is total flux × energy percentage
                                    actual_total_flux = flux_actual[label]
                                    actual_energy_frac = energy_groups[label][energy_group]
                                    actual_val = actual_total_flux * actual_energy_frac
                                else:
                                    # No energy group data - ERROR OUT as requested
                                    raise ValueError(f"No energy group data available for {label}. Cannot calculate {energy_group} flux without energy group percentages.")

                                # Extract label number
                                label_num = int(label.split('_')[1])

                                result[f'I_{label_num}_{energy_group}_real'] = actual_val
                                result[f'I_{label_num}_{energy_group}_predicted'] = pred_val
                                # SIGNED MAPE: positive = overprediction, negative = underprediction
                                result[f'I_{label_num}_{energy_group}_rel_error'] = ((pred_val - actual_val) / actual_val * 100) if actual_val != 0 else 0

                                flux_pred_list.append(pred_val)
                                flux_actual_list.append(actual_val)

                        # Calculate average metrics
                        if flux_pred_list and flux_actual_list:
                            avg_pred = np.mean(flux_pred_list)
                            avg_actual = np.mean(flux_actual_list)

                            result[f'avg_{energy_group}_flux_real'] = avg_actual
                            result[f'avg_{energy_group}_flux_predicted'] = avg_pred
                            # SIGNED MAPE: positive = overprediction, negative = underprediction
                            result[f'avg_{energy_group}_flux_rel_error'] = ((avg_pred - avg_actual) / avg_actual * 100) if avg_actual != 0 else 0
                            result[f'mape_{energy_group}_flux'] = np.mean([((p - a) / a) * 100 for p, a in zip(flux_pred_list, flux_actual_list) if a != 0])

                            # Also add a general MAPE for compatibility with Excel reporter
                            result['mape_flux'] = result[f'mape_{energy_group}_flux']

                elif model_info['model_type'] == 'keff':
                    # Check if model needs scaling (for backward compatibility with raw models)
                    if model_wrapper is None:
                        # Old raw model approach
                        model = model_data.get('model')
                        if 'scaler' in model_data:
                            features = model_data['scaler'].transform(features)
                        keff_pred = model.predict(features)
                    else:
                        # New wrapper approach - use the wrapper's predict method
                        keff_pred = model_wrapper.predict_keff(features)

                    if hasattr(keff_pred, '__len__'):
                        keff_pred = keff_pred[0]

                    result['keff_real'] = keff_actual
                    result['keff_predicted'] = keff_pred
                    # SIGNED MAPE: positive = overprediction, negative = underprediction
                    result['keff_rel_error'] = ((keff_pred - keff_actual) / keff_actual * 100) if keff_actual != 0 else 0

                    # Store MAPE for internal use
                    result['mape_keff'] = result['keff_rel_error']

                results.append(result)

            except Exception as e:
                print(f"\nError testing {model_info['model_class']} {model_info['model_type']}: {e}")
                import traceback
                traceback.print_exc()

        return results

# Complete replacement for test_file() method in model_tester.py

    def test_file(self, test_file_path, training_file_path="ML/data/train.txt", show_match_details=False,
                save_intermediate_results=None):
        """Test all models on a test file

        Args:
            test_file_path: Path to test data
            training_file_path: Path to training data
            show_match_details: Whether to show detailed matching info
            save_intermediate_results: If provided, function to save intermediate results
                                    (used for saving total flux results for bin models)
        """
        print(f"\nLoading test data from {test_file_path}...")

        # Load training data for comparison
        self.load_training_data(training_file_path)

        # Parse test data - try to get energy groups too
        try:
            # Try parsing with all data including energy groups
            parse_result = parse_reactor_data(test_file_path)
            if len(parse_result) == 5:
                # New format with energy groups
                lattices, flux_data, k_effectives, descriptions, energy_groups = parse_result
                print(f"Successfully loaded energy group data from test file")
            elif len(parse_result) == 4:
                # Format with descriptions but no energy groups
                lattices, flux_data, k_effectives, descriptions = parse_result
                energy_groups = [{}] * len(lattices)
                print(f"Warning: No energy group data in test file")
            else:
                # Old format without descriptions
                lattices, flux_data, k_effectives = parse_result
                descriptions = [""] * len(lattices)
                energy_groups = [{}] * len(lattices)
                print(f"Warning: Using old format, no energy groups")
        except Exception as e:
            print(f"Error parsing test data: {e}")
            # Fallback to old format
            lattices, flux_data, k_effectives = parse_reactor_data(test_file_path)
            descriptions = [""] * len(lattices)
            energy_groups = [{}] * len(lattices)

        print(f"Found {len(lattices)} test configurations")

        # Group models by type for ordered testing - MUST DEFINE ALL VARIABLES!
        total_flux_models = [m for m in self.available_models
                            if m['model_type'] == 'flux' and m.get('flux_mode', 'total') == 'total']
        energy_flux_models = [m for m in self.available_models
                            if m['model_type'] == 'flux' and m.get('flux_mode') == 'energy']
        bin_flux_models = [m for m in self.available_models
                        if m['model_type'] == 'flux' and m.get('flux_mode') == 'bin']
        thermal_only_models = [m for m in self.available_models
                            if m['model_type'] == 'flux' and m.get('flux_mode') == 'thermal_only']
        epithermal_only_models = [m for m in self.available_models
                                if m['model_type'] == 'flux' and m.get('flux_mode') == 'epithermal_only']
        fast_only_models = [m for m in self.available_models
                        if m['model_type'] == 'flux' and m.get('flux_mode') == 'fast_only']
        keff_models = [m for m in self.available_models if m['model_type'] == 'keff']

        # Debug print to see what models we have
        print(f"\nModel counts by type:")
        print(f"  Total flux: {len(total_flux_models)}")
        print(f"  Energy flux: {len(energy_flux_models)}")
        print(f"  Bin flux: {len(bin_flux_models)}")
        print(f"  Thermal only: {len(thermal_only_models)}")
        print(f"  Epithermal only: {len(epithermal_only_models)}")
        print(f"  Fast only: {len(fast_only_models)}")
        print(f"  K-eff: {len(keff_models)}")

        all_results = []
        match_count = 0

        # Process total flux models first
        if total_flux_models:
            print(f"\nTesting {len(total_flux_models)} total flux model(s)...")
            self.available_models = total_flux_models

            for i, (lattice, flux, keff, desc, energy_group) in enumerate(zip(lattices, flux_data, k_effectives, descriptions, energy_groups)):
                print(f"\rTesting configuration {i+1}/{len(lattices)}...", end='')

                # Check if this configuration was in training set
                in_training = self.is_in_training_set(lattice)
                if in_training:
                    match_count += 1

                # Optionally show detailed match information
                if show_match_details and in_training:
                    match_found, train_idx, transform = self.find_matching_training_config(lattice)
                    print(f"\n  Config {i+1} matches training config {train_idx+1} with transform: {transform}")

                results = self.test_single_configuration(lattice, flux, keff, i, desc, in_training, energy_group)
                all_results.extend(results)

            # Save intermediate results if we have bin models coming up
            if bin_flux_models and save_intermediate_results:
                print("\n\nSaving total flux results for bin model calculations...")
                total_flux_path = save_intermediate_results(all_results.copy())
                if total_flux_path:
                    self.load_total_flux_results(total_flux_path)

        # Process energy flux models
        if energy_flux_models:
            print(f"\n\nTesting {len(energy_flux_models)} energy flux model(s)...")
            self.available_models = energy_flux_models

            for i, (lattice, flux, keff, desc, energy_group) in enumerate(zip(lattices, flux_data, k_effectives, descriptions, energy_groups)):
                print(f"\rTesting configuration {i+1}/{len(lattices)}...", end='')

                # Check if this configuration was in training set (only count once)
                in_training = self.is_in_training_set(lattice)

                results = self.test_single_configuration(lattice, flux, keff, i, desc, in_training, energy_group)
                all_results.extend(results)

        # Process bin flux models
        if bin_flux_models:
            print(f"\n\nTesting {len(bin_flux_models)} bin flux model(s)...")
            print(f"(Each model predicts 3 energy bins × 4 positions = 12 outputs)")

            # Always ask for total flux results for bin models
            print("\nBin models require total flux data to calculate absolute fluxes.")
            total_flux_path = input("What is the path file for total flux Excel results? ").strip()
            if total_flux_path:
                if self.load_total_flux_results(total_flux_path):
                    print("Successfully loaded total flux data.")
                else:
                    print("Warning: Failed to load total flux data. Results will be incomplete.")
            else:
                print("Warning: No total flux data provided. Bin model results will be incomplete.")

            self.available_models = bin_flux_models

            for i, (lattice, flux, keff, desc, energy_group) in enumerate(zip(lattices, flux_data, k_effectives, descriptions, energy_groups)):
                print(f"\rTesting configuration {i+1}/{len(lattices)}...", end='')

                # Check if this configuration was in training set (only count once)
                in_training = self.is_in_training_set(lattice)

                results = self.test_single_configuration(lattice, flux, keff, i, desc, in_training, energy_group)
                all_results.extend(results)

        # Process single energy models - THIS WAS MISSING!
        for energy_type, energy_models in [('thermal', thermal_only_models),
                                        ('epithermal', epithermal_only_models),
                                        ('fast', fast_only_models)]:
            if energy_models:
                print(f"\n\nTesting {len(energy_models)} {energy_type} only flux model(s)...")
                print(f"(Each model predicts {energy_type} flux for 4 positions)")
                self.available_models = energy_models

                for i, (lattice, flux, keff, desc, energy_group) in enumerate(zip(lattices, flux_data, k_effectives, descriptions, energy_groups)):
                    print(f"\rTesting configuration {i+1}/{len(lattices)}...", end='', flush=True)

                    # Check if this configuration was in training set
                    in_training = self.is_in_training_set(lattice)

                    # Check if we have energy group data
                    if not energy_group:
                        print(f"\nWarning: No energy group data for config {i+1}, skipping {energy_type} only test")
                        continue

                    results = self.test_single_configuration(lattice, flux, keff, i, desc, in_training, energy_group)
                    all_results.extend(results)

        # Process k-eff models
        if keff_models:
            print(f"\n\nTesting {len(keff_models)} k-eff model(s)...")
            self.available_models = keff_models

            for i, (lattice, flux, keff, desc, energy_group) in enumerate(zip(lattices, flux_data, k_effectives, descriptions, energy_groups)):
                print(f"\rTesting configuration {i+1}/{len(lattices)}...", end='')

                # Check if this configuration was in training set (only count once)
                in_training = self.is_in_training_set(lattice)

                results = self.test_single_configuration(lattice, flux, keff, i, desc, in_training, energy_group)
                all_results.extend(results)

        print(f"\n\nTesting complete!")
        print(f"\nSummary: {match_count}/{len(lattices)} test configurations were seen during training (considering symmetries)")

        return all_results
