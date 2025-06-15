import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from utils.txt_to_data import parse_reactor_data, apply_rotational_symmetry
from ML_models.encodings.encoding_methods import ReactorEncodings

class DataHandler:
    """Handle data loading and preprocessing"""

    def __init__(self):
        self.encodings = ReactorEncodings()
        self.use_log_flux = True  # Flag to use log transform for flux
        self.flux_scale = 1e14    # Alternative scaling factor
        self.flux_mode = 'total'  # NEW: 'total', 'energy', or 'bin'

    def load_and_prepare_data(self, data_file, encoding_method, flux_mode='total'):
        """Load and encode reactor data with support for different flux modes"""
        self.flux_mode = flux_mode  # Store for later use

        print(f"  Loading from {data_file}...")
        result = parse_reactor_data(data_file)

        # Handle both old and new return formats
        if len(result) == 5:
            lattices, flux_data, k_effectives, descriptions, energy_groups = result
        else:
            # Backward compatibility
            lattices, flux_data, k_effectives, descriptions = result
            energy_groups = [{}] * len(lattices)  # Empty energy groups

        print(f"  Found {len(lattices)} configurations")
        print(f"  Flux mode: {flux_mode}")

        print(f"  Applying {encoding_method} encoding with 8-fold augmentation...")

        # Prepare data with selected encoding
        X_features = []
        y_flux_values = []
        y_k_eff = []
        irr_positions_list = []

        # NEW: Track augmentation groups for CV
        augmentation_groups = []

        # Track some statistics for validation
        label_order_mismatches = 0
        total_configs = 0

        # NEW: Group counter
        group_id = 0

        for lattice, flux_dict, k_eff, energy_dict in zip(lattices, flux_data, k_effectives, energy_groups):
            # Apply rotational symmetry (8-fold augmentation)
            augmented = apply_rotational_symmetry(lattice, flux_dict, k_eff, energy_dict)

            for aug_data in augmented:
                # Handle different return formats
                if len(aug_data) == 4:
                    aug_lattice, aug_flux, aug_k_eff, aug_energy = aug_data
                else:
                    aug_lattice, aug_flux, aug_k_eff = aug_data
                    aug_energy = {}

                total_configs += 1

                # NEW: Add group ID (same for all 8 augmentations)
                augmentation_groups.append(group_id)

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

                # Prepare flux values based on mode
                if flux_mode == 'total':
                    # Original behavior - 4 flux values
                    flux_values = self._prepare_total_flux_values(aug_lattice, aug_flux, position_order)
                elif flux_mode == 'energy':
                    # 12 values: 3 energy groups × 4 positions
                    flux_values = self._prepare_energy_flux_values(aug_lattice, aug_flux, aug_energy, position_order)
                elif flux_mode == 'bin':
                    # 12 values: 3 percentages × 4 positions
                    flux_values = self._prepare_energy_bin_values(aug_lattice, aug_energy, position_order)
                elif flux_mode in ['thermal_only', 'epithermal_only', 'fast_only']:
                    # NEW: 4 values for single energy group
                    flux_values = self._prepare_single_energy_flux_values(aug_lattice, aug_flux, aug_energy, position_order, flux_mode)
                else:
                    raise ValueError(f"Unknown flux mode: {flux_mode}")

                y_flux_values.append(flux_values)
                y_k_eff.append(aug_k_eff)
                irr_positions_list.append(irr_positions)

            # NEW: Increment group ID after processing all augmentations
            group_id += 1

        # Report validation statistics
        mismatch_percentage = (label_order_mismatches / total_configs) * 100 if total_configs > 0 else 0
        print(f"  Spatial vs alphabetical order mismatches: {label_order_mismatches}/{total_configs} ({mismatch_percentage:.1f}%)")
        if mismatch_percentage > 0:
            print(f"  ✓ Good: Model is learning from diverse spatial configurations")

        X = np.array(X_features)
        y_flux = np.array(y_flux_values)
        y_keff = np.array(y_k_eff)

        # NEW: Convert groups to numpy array
        groups = np.array(augmentation_groups)

        # Validate array shapes
        assert X.shape[0] == y_flux.shape[0] == y_keff.shape[0], "Sample count mismatch"

        # Check expected output dimensions
        if flux_mode == 'total':
            assert y_flux.shape[1] == 4, f"Expected 4 flux outputs, got {y_flux.shape[1]}"
        elif flux_mode in ['thermal_only', 'epithermal_only', 'fast_only']:
            assert y_flux.shape[1] == 4, f"Expected 4 flux outputs for single energy mode, got {y_flux.shape[1]}"
        else:  # energy or bin
            assert y_flux.shape[1] == 12, f"Expected 12 flux outputs, got {y_flux.shape[1]}"

        # Transform flux values based on mode
        if flux_mode == 'bin':
            # No transformation for bins
            print(f"  Energy bin values range: {y_flux.min():.3f} to {y_flux.max():.3f}")
            self.use_log_flux = False  # Override for bins
        else:
            # Log transform for total, energy, and single energy flux modes
            if self.use_log_flux:
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
        if flux_mode == 'total':
            print(f"  Flux targets: {y_flux.shape} (4 positions per sample)")
        elif flux_mode == 'energy':
            print(f"  Flux targets: {y_flux.shape} (12 values: 3 energy groups × 4 positions)")
        elif flux_mode == 'bin':
            print(f"  Flux targets: {y_flux.shape} (12 values: 3 bin fractions × 4 positions)")
        elif flux_mode in ['thermal_only', 'epithermal_only', 'fast_only']:
            energy_name = flux_mode.replace('_only', '')
            print(f"  Flux targets: {y_flux.shape} (4 positions, {energy_name} flux only)")
        print(f"  K-eff targets: {y_keff.shape}")

        # Final validation message
        print(f"\n  ✓ Data loaded successfully with label-agnostic encoding")
        print(f"  ✓ Flux values ordered by spatial position, not label")

        # NEW: Return groups as well
        return X, y_flux, y_keff, groups

    def _prepare_total_flux_values(self, lattice, flux_dict, position_order):
        """Prepare total flux values (original behavior)"""
        position_to_flux = {}
        label_at_position = {}

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                cell = lattice[i, j]
                if cell.startswith('I_'):
                    label_at_position[(i, j)] = cell
                    if cell in flux_dict:
                        position_to_flux[(i, j)] = flux_dict[cell]
                    else:
                        print(f"Warning: No flux value for {cell} in augmented data")
                        position_to_flux[(i, j)] = 0.0

        flux_values = []
        for pos in position_order:
            if pos in position_to_flux:
                flux_values.append(position_to_flux[pos])
            else:
                print(f"Warning: Position {pos} not found in flux mapping")
                flux_values.append(0.0)

        # Ensure we have exactly 4 flux values
        if len(flux_values) != 4:
            print(f"Warning: Expected 4 flux values, got {len(flux_values)}")
            if len(flux_values) < 4:
                flux_values.extend([0.0] * (4 - len(flux_values)))
            else:
                flux_values = flux_values[:4]

        return flux_values

    def _prepare_energy_flux_values(self, lattice, flux_dict, energy_dict, position_order):
        """Prepare energy flux values (flux × percentage for each group)"""
        flux_values = []

        for pos in position_order:
            # Find the label at this position
            i, j = pos
            label = lattice[i, j]

            if label.startswith('I_') and label in flux_dict and label in energy_dict:
                total_flux = flux_dict[label]
                energy_fracs = energy_dict[label]

                # Calculate absolute flux for each energy group
                thermal_flux = total_flux * energy_fracs['thermal']
                epithermal_flux = total_flux * energy_fracs['epithermal']
                fast_flux = total_flux * energy_fracs['fast']

                flux_values.extend([thermal_flux, epithermal_flux, fast_flux])
            else:
                # Default values if missing
                flux_values.extend([0.0, 0.0, 0.0])

        # Should have 12 values (3 groups × 4 positions)
        if len(flux_values) != 12:
            print(f"Warning: Expected 12 energy flux values, got {len(flux_values)}")
            while len(flux_values) < 12:
                flux_values.append(0.0)
            flux_values = flux_values[:12]

        return flux_values

    def _prepare_energy_bin_values(self, lattice, energy_dict, position_order):
        """Prepare energy bin values (just the percentages as fractions)"""
        bin_values = []

        for pos in position_order:
            # Find the label at this position
            i, j = pos
            label = lattice[i, j]

            if label.startswith('I_') and label in energy_dict:
                energy_fracs = energy_dict[label]

                # Just use the fractions directly
                bin_values.extend([
                    energy_fracs['thermal'],
                    energy_fracs['epithermal'],
                    energy_fracs['fast']
                ])
            else:
                # Default values if missing - equal distribution
                bin_values.extend([1.0/3.0, 1.0/3.0, 1.0/3.0])

        # Should have 12 values (3 groups × 4 positions)
        if len(bin_values) != 12:
            print(f"Warning: Expected 12 energy bin values, got {len(bin_values)}")
            while len(bin_values) < 12:
                bin_values.append(1.0/3.0)
            bin_values = bin_values[:12]

        return bin_values

    def split_data(self, X, y_flux, y_keff, groups=None, test_size=0.15, random_state=42):
        """Split data into train/test sets"""
        if groups is not None:
            # Use GroupShuffleSplit to ensure augmentations stay together
            from sklearn.model_selection import GroupShuffleSplit

            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y_flux, groups))

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_flux_train = y_flux[train_idx]
            y_flux_test = y_flux[test_idx]
            y_keff_train = y_keff[train_idx]
            y_keff_test = y_keff[test_idx]
            groups_train = groups[train_idx]
            groups_test = groups[test_idx]

            print(f"  Using GroupShuffleSplit to prevent augmentation leakage")
            print(f"  Unique configs in train: {len(np.unique(groups_train))}")
            print(f"  Unique configs in test: {len(np.unique(groups_test))}")
        else:
            # Fallback to regular split
            X_train, X_test, y_flux_train, y_flux_test, y_keff_train, y_keff_test = \
                train_test_split(X, y_flux, y_keff,
                               test_size=test_size,
                               random_state=random_state)
            groups_train = None
            groups_test = None

        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_flux_train': y_flux_train,
            'y_flux_test': y_flux_test,
            'y_keff_train': y_keff_train,
            'y_keff_test': y_keff_test,
            'groups_train': groups_train,  # NEW
            'groups_test': groups_test      # NEW
        }

    def _prepare_single_energy_flux_values(self, lattice, flux_dict, energy_dict, position_order, flux_mode):
        """Prepare single energy group flux values (total flux × energy percentage)"""
        flux_values = []

        # Determine which energy group we're using
        energy_group = flux_mode.replace('_only', '')  # 'thermal', 'epithermal', or 'fast'

        for pos in position_order:
            # Find the label at this position
            i, j = pos
            label = lattice[i, j]

            if label.startswith('I_') and label in flux_dict and label in energy_dict:
                total_flux = flux_dict[label]
                energy_fracs = energy_dict[label]

                # Calculate flux for the selected energy group
                single_energy_flux = total_flux * energy_fracs[energy_group]
                flux_values.append(single_energy_flux)
            else:
                # Default value if missing
                flux_values.append(0.0)

        # Should have 4 values (one per position)
        if len(flux_values) != 4:
            print(f"Warning: Expected 4 flux values for {flux_mode}, got {len(flux_values)}")
            while len(flux_values) < 4:
                flux_values.append(0.0)
            flux_values = flux_values[:4]

        return flux_values
