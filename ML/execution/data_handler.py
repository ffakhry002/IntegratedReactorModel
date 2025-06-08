import numpy as np
from sklearn.model_selection import train_test_split
from utils.txt_to_data import parse_reactor_data, apply_rotational_symmetry
from ML_models.encodings.encoding_methods import ReactorEncodings

class DataHandler:
    """Handle data loading and preprocessing"""

    def __init__(self):
        self.encodings = ReactorEncodings()
        self.use_log_flux = True  # Flag to use log transform for flux
        self.flux_scale = 1e14    # Alternative scaling factor

    def load_and_prepare_data(self, data_file, encoding_method):
        """Load and encode reactor data"""
        print(f"  Loading from {data_file}...")
        lattices, flux_data, k_effectives, descriptions = parse_reactor_data(data_file)
        print(f"  Found {len(lattices)} configurations")

        print(f"  Applying {encoding_method} encoding with 8-fold augmentation...")

        # Prepare data with selected encoding
        X_features = []
        y_flux_values = []
        y_k_eff = []
        irr_positions_list = []

        # Track some statistics for validation
        label_order_mismatches = 0
        total_configs = 0

        for lattice, flux_dict, k_eff in zip(lattices, flux_data, k_effectives):
            # Apply rotational symmetry (8-fold augmentation)
            augmented = apply_rotational_symmetry(lattice, flux_dict, k_eff)

            for aug_lattice, aug_flux, aug_k_eff in augmented:
                total_configs += 1

                # Encode using selected method - now returns position order
                if encoding_method == 'one_hot':
                    feature_vec, irr_positions, position_order = self.encodings.one_hot_encoding(aug_lattice)
                elif encoding_method == 'categorical':
                    feature_vec, irr_positions, position_order = self.encodings.categorical_encoding(aug_lattice)
                elif encoding_method == 'physics':
                    feature_vec, irr_positions, position_order = self.encodings.physics_based_encoding(aug_lattice)
                elif encoding_method == 'spatial':
                    feature_vec, irr_positions, position_order = self.encodings.spatial_convolution_encoding(aug_lattice)
                elif encoding_method == 'graph':
                    feature_vec, irr_positions, position_order = self.encodings.graph_based_encoding(aug_lattice)
                else:
                    raise ValueError(f"Unknown encoding method: {encoding_method}")

                X_features.append(feature_vec)

                # CRITICAL: Collect flux values based on SPATIAL position order
                # This ensures the model learns position->flux relationships

                # Step 1: Create a mapping from position to flux value
                position_to_flux = {}
                label_at_position = {}

                for i in range(aug_lattice.shape[0]):
                    for j in range(aug_lattice.shape[1]):
                        cell = aug_lattice[i, j]
                        if cell.startswith('I_'):
                            # Store what label is at this position
                            label_at_position[(i, j)] = cell
                            # Store the flux value for this position
                            if cell in aug_flux:
                                position_to_flux[(i, j)] = aug_flux[cell]
                            else:
                                print(f"Warning: No flux value for {cell} in augmented data")
                                position_to_flux[(i, j)] = 0.0

                # Step 2: Collect flux values in SPATIAL order (not alphabetical!)
                # position_order is sorted by (row, col), ensuring consistent spatial ordering
                flux_values = []
                spatial_labels = []  # For validation

                for pos in position_order:
                    if pos in position_to_flux:
                        flux_values.append(position_to_flux[pos])
                        if pos in label_at_position:
                            spatial_labels.append(label_at_position[pos])
                    else:
                        # This shouldn't happen, but handle gracefully
                        print(f"Warning: Position {pos} not found in flux mapping")
                        flux_values.append(0.0)

                # Validation: Check if spatial order differs from alphabetical
                if spatial_labels:
                    alphabetical_labels = sorted(spatial_labels)
                    if spatial_labels != alphabetical_labels:
                        label_order_mismatches += 1

                # Ensure we have exactly 4 flux values
                if len(flux_values) != 4:
                    print(f"Warning: Expected 4 flux values, got {len(flux_values)}")
                    # Pad or truncate to 4 values
                    if len(flux_values) < 4:
                        flux_values.extend([0.0] * (4 - len(flux_values)))
                    else:
                        flux_values = flux_values[:4]

                y_flux_values.append(flux_values)
                y_k_eff.append(aug_k_eff)
                irr_positions_list.append(irr_positions)

        # Report validation statistics
        mismatch_percentage = (label_order_mismatches / total_configs) * 100 if total_configs > 0 else 0
        print(f"  Spatial vs alphabetical order mismatches: {label_order_mismatches}/{total_configs} ({mismatch_percentage:.1f}%)")
        if mismatch_percentage > 0:
            print(f"  ✓ Good: Model is learning from diverse spatial configurations")

        X = np.array(X_features)
        y_flux = np.array(y_flux_values)
        y_keff = np.array(y_k_eff)

        # Validate array shapes
        assert X.shape[0] == y_flux.shape[0] == y_keff.shape[0], "Sample count mismatch"
        assert y_flux.shape[1] == 4, f"Expected 4 flux outputs, got {y_flux.shape[1]}"

        # Transform flux values to log scale or simple scaling
        if self.use_log_flux:
            # Log transform for flux
            y_flux_original = y_flux.copy()  # Keep for reference
            y_flux = np.log10(y_flux + 1e-10)  # Add small value to avoid log(0)
            print(f"  Flux values log-transformed. Range: {y_flux.min():.2f} to {y_flux.max():.2f}")
            print(f"  Original flux range: {y_flux_original.min():.2e} to {y_flux_original.max():.2e}")
        else:
            # Alternative: simple scaling
            y_flux = y_flux / self.flux_scale
            print(f"  Flux values scaled by {self.flux_scale:.0e}. Range: {y_flux.min():.2f} to {y_flux.max():.2f}")

        print(f"  After augmentation: {X.shape[0]} samples")
        print(f"  Feature shape: {X.shape}")
        print(f"  Flux targets: {y_flux.shape} (4 positions per sample)")
        print(f"  K-eff targets: {y_keff.shape}")

        # Final validation message
        print(f"\n  ✓ Data loaded successfully with label-agnostic encoding")
        print(f"  ✓ Flux values ordered by spatial position, not label")

        return X, y_flux, y_keff

    def split_data(self, X, y_flux, y_keff, test_size=0.15, random_state=42):
        """Split data into train/test sets"""
        X_train, X_test, y_flux_train, y_flux_test, y_keff_train, y_keff_test = \
            train_test_split(X, y_flux, y_keff,
                           test_size=test_size,
                           random_state=random_state)

        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_flux_train': y_flux_train,
            'y_flux_test': y_flux_test,
            'y_keff_train': y_keff_train,
            'y_keff_test': y_keff_test
        }
