"""
Base class for all sampling methods with distance caching and optimization.
FIXED: Lattice Euclidean/Manhattan distances now use canonical forms for symmetry awareness
"""

import numpy as np
import pickle
import json
import os
import threading
from typing import List, Tuple, Dict, Set, Optional, Callable
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from .symmetry_utils import ALL_SYMMETRIES, are_configurations_symmetric, get_canonical_form
from .cache import EfficientDistanceCache
from .utils import UnifiedDiscretization
from abc import ABC, abstractmethod
from sklearn.preprocessing import KBinsDiscretizer

# Get the script directory (parent of sampling_methods)
SCRIPT_DIR = Path(__file__).parent.parent.absolute()

class BaseSampler:
    """Base class for all sampling methods with caching and optimization."""

    def __init__(self, use_6x6_restriction=False, selected_parameters=None):
        # Only set default method_name if not already set by child class
        if not hasattr(self, 'method_name'):
            self.method_name = "base"

        # Configuration restriction setting
        self.use_6x6_restriction = use_6x6_restriction

        # Store selected parameters (None means use all available)
        self.selected_parameters = selected_parameters

        self._cache_lock = threading.Lock()
        # Initialize data attributes
        self.configurations = None
        self.irradiation_sets = None
        self.physics_parameters = None
        self.feature_matrix = None
        self.feature_matrix_normalized = None
        self.scaler = StandardScaler()

        # Define feature names for physics parameters (will be updated after loading data)
        self.feature_names = [
            'avg_distance_from_core_center',
            'min_inter_position_distance',
            'clustering_coefficient',
            'symmetry_balance',
            'local_fuel_density'
        ]

        # Initialize efficient distance cache
        self.distance_cache = EfficientDistanceCache(max_cache_size=50000)

        # Additional optimization structures
        self.position_sets = None
        self.coords_cache = {}
        self.canonical_forms = {}  # Cache for canonical forms
        self.canonical_coords_cache = {}  # NEW: Cache for canonical form coordinates

        # Cache for precomputed discretized features (geometric Jaccard)
        self.discretized_features_cache = None

        # Initialize unified discretizer
        self._discretizer = UnifiedDiscretization(n_bins=15)

        # Load data
        self.load_data()

    def _precompute_optimization_data(self):
        """Precompute data structures for optimization."""
        print("Precomputing optimization data...")

        # Precompute position sets for Jaccard calculations
        if self.position_sets is None and self.irradiation_sets is not None:
            self.position_sets = [set(positions) for positions in self.irradiation_sets]

        # Precompute coordinate arrays for lattice distance calculations
        if self.irradiation_sets is not None:
            for idx, positions in enumerate(self.irradiation_sets):
                if idx not in self.coords_cache:
                    self.coords_cache[idx] = np.array([(i + 0.5, j + 0.5) for i, j in positions])

        # Precompute canonical forms for symmetry-aware methods (only for lattice methods)
        if 'lattice' in self.method_name and self.irradiation_sets is not None:
            print("  Computing canonical forms for symmetry-aware comparisons...")
            for idx, positions in enumerate(self.irradiation_sets):
                if idx not in self.canonical_forms:
                    self.canonical_forms[idx] = get_canonical_form(positions)
                    # FIXED: Also cache coordinates for canonical forms
                    self.canonical_coords_cache[idx] = np.array([(i + 0.5, j + 0.5)
                                                                for i, j in self.canonical_forms[idx]])

    def _get_cache_key(self, idx1: int, idx2: int) -> Tuple[int, int]:
        """Generate symmetric cache key for two indices."""
        return tuple(sorted([idx1, idx2]))

    def get_cached_euclidean_distance(self, idx1: int, idx2: int) -> float:
        """Get cached Euclidean distance or calculate and cache if not exists."""
        def compute_func(i1, i2):
            return np.linalg.norm(
                self.feature_matrix_normalized[i1] -
                self.feature_matrix_normalized[i2]
            )

        return self.distance_cache.get_distance('euclidean', idx1, idx2, compute_func)

    def get_cached_manhattan_distance(self, idx1: int, idx2: int) -> float:
        """Get cached Manhattan distance or calculate and cache if not exists."""
        def compute_func(i1, i2):
            return np.sum(np.abs(
                self.feature_matrix_normalized[i1] -
                self.feature_matrix_normalized[i2]
            ))

        return self.distance_cache.get_distance('manhattan', idx1, idx2, compute_func)

    def get_cached_jaccard_distance(self, idx1: int, idx2: int) -> float:
        """Get cached Jaccard distance or calculate and cache if not exists."""
        def compute_func(i1, i2):
            # For geometric methods, use continuous Jaccard
            if 'geometric' in self.method_name:
                feat1 = self.feature_matrix_normalized[i1]
                feat2 = self.feature_matrix_normalized[i2]

                # Ensure features are non-negative by shifting
                min_val = min(feat1.min(), feat2.min(), 0)
                feat1_shifted = feat1 - min_val
                feat2_shifted = feat2 - min_val

                # Continuous Jaccard calculation
                numerator = np.sum(np.minimum(feat1_shifted, feat2_shifted))
                denominator = np.sum(np.maximum(feat1_shifted, feat2_shifted))

                if denominator == 0:
                    return 0.0

                jaccard_similarity = numerator / denominator
                return 1.0 - jaccard_similarity
            else:
                # For lattice methods, use position sets as before
                if self.position_sets is None:
                    self._precompute_optimization_data()

                distance = self.calculate_jaccard_distance(
                    self.position_sets[i1],
                    self.position_sets[i2]
                )

                return distance

        return self.distance_cache.get_distance('jaccard', idx1, idx2, compute_func)

    def get_cached_lattice_euclidean_distance(self, idx1: int, idx2: int) -> float:
        """Get cached lattice Euclidean distance or calculate and cache if not exists.
        FIXED: Now uses canonical forms for symmetry-aware comparison."""
        def compute_func(i1, i2):
            # FIXED: Use canonical form coordinates instead of raw coordinates
            if i1 not in self.canonical_coords_cache or i2 not in self.canonical_coords_cache:
                self._precompute_optimization_data()

            return self._calculate_greedy_euclidean_distance(
                self.canonical_coords_cache[i1],  # Use canonical forms
                self.canonical_coords_cache[i2]   # Use canonical forms
            )

        return self.distance_cache.get_distance('lattice_euclidean', idx1, idx2, compute_func)

    def get_cached_lattice_manhattan_distance(self, idx1: int, idx2: int) -> float:
        """Get cached lattice Manhattan distance or calculate and cache if not exists.
        FIXED: Now uses canonical forms for symmetry-aware comparison."""
        def compute_func(i1, i2):
            # FIXED: Use canonical form coordinates instead of raw coordinates
            if i1 not in self.canonical_coords_cache or i2 not in self.canonical_coords_cache:
                self._precompute_optimization_data()

            return self._calculate_greedy_manhattan_distance(
                self.canonical_coords_cache[i1],  # Use canonical forms
                self.canonical_coords_cache[i2]   # Use canonical forms
            )

        return self.distance_cache.get_distance('lattice_manhattan', idx1, idx2, compute_func)

    def _calculate_greedy_euclidean_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate Euclidean distance using greedy matching."""
        distances = []
        used_indices = set()

        for coord1 in coords1:
            min_dist = float('inf')
            best_idx = -1

            for idx, coord2 in enumerate(coords2):
                if idx not in used_indices:
                    dist = np.linalg.norm(coord1 - coord2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx

            if best_idx != -1:
                distances.append(min_dist)
                used_indices.add(best_idx)

        return np.mean(distances) if distances else 0.0

    def _calculate_greedy_manhattan_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate Manhattan distance using greedy matching."""
        distances = []
        used_indices = set()

        for coord1 in coords1:
            min_dist = float('inf')
            best_idx = -1

            for idx, coord2 in enumerate(coords2):
                if idx not in used_indices:
                    dist = np.sum(np.abs(coord1 - coord2))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx

            if best_idx != -1:
                distances.append(min_dist)
                used_indices.add(best_idx)

        return np.mean(distances) if distances else 0.0

    def load_data(self):
        """Load configurations and physics parameters."""
        # Determine which configuration file to load based on method type
        # Geometric methods should use full set, lattice methods use reduced set

        # Add suffix for 6x6 restriction
        suffix = "_6x6" if self.use_6x6_restriction else ""

        if 'geometric' in self.method_name or self.method_name in ['lhs', 'sobol', 'halton']:
            # Load FULL configurations for geometric methods
            try:
                full_config_path = SCRIPT_DIR / f'output/data/all_configurations_before_symmetry{suffix}.pkl'
                with open(full_config_path, 'rb') as f:
                    config_data = pickle.load(f)
                restriction_info = " (6x6 restricted)" if self.use_6x6_restriction else ""
                print(f"  Loaded FULL configuration set{restriction_info} ({len(config_data['configurations'])} configs) for {self.method_name}")
                using_full_set = True
            except FileNotFoundError:
                # Fallback to optimized file if full file doesn't exist
                optimized_path = SCRIPT_DIR / f'output/data/core_configurations_optimized{suffix}.pkl'
                with open(optimized_path, 'rb') as f:
                    config_data = pickle.load(f)
                print(f"  Warning: Full config file not found, using optimized set")
                using_full_set = False
        else:
            # Load symmetry-reduced configurations for lattice methods
            optimized_path = SCRIPT_DIR / f'output/data/core_configurations_optimized{suffix}.pkl'
            with open(optimized_path, 'rb') as f:
                config_data = pickle.load(f)
            restriction_info = " (6x6 restricted)" if self.use_6x6_restriction else ""
            print(f"  Loaded symmetry-reduced set{restriction_info} ({len(config_data['configurations'])} configs) for {self.method_name}")
            using_full_set = False

        self.configurations = config_data['configurations']
        self.irradiation_sets = config_data['irradiation_sets']

        # Load physics parameters
        # Note: Physics parameters should be calculated for 6x6 restricted configs too
        # For now, we'll use the same physics parameter files
        # TODO: Generate separate physics parameter files for 6x6 restricted configs

        # For geometric methods (random, halton, sobol, lhs), we need the FULL parameter set
        if 'geometric' in self.method_name or self.method_name in ['lhs', 'sobol', 'halton']:
            # Try to load the full set first
            try:
                full_params_path = SCRIPT_DIR / f'output/data/physics_parameters_full{suffix}.pkl'
                with open(full_params_path, 'rb') as f:
                    all_parameters = pickle.load(f)
                self.physics_parameters = all_parameters['parameters']
                print(f"  Loaded physics parameters for FULL set ({len(self.physics_parameters)} params)")
            except FileNotFoundError:
                print("  Warning: Full physics parameters not found, falling back to reduced set")
                # Fall back to reduced set
                reduced_params_path = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'
                with open(reduced_params_path, 'rb') as f:
                    all_parameters = pickle.load(f)
                self.physics_parameters = all_parameters['parameters']
                print(f"  Loaded physics parameters for reduced set ({len(self.physics_parameters)} params)")
        else:
            # For lattice-based methods, we only need the reduced set
            reduced_params_path = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'
            with open(reduced_params_path, 'rb') as f:
                all_parameters = pickle.load(f)
            self.physics_parameters = all_parameters['parameters']
            print(f"  Loaded physics parameters for reduced set ({len(self.physics_parameters)} params)")

        # Verify lengths match
        if len(self.configurations) != len(self.physics_parameters):
            print(f"  ERROR: Configuration count ({len(self.configurations)}) != physics param count ({len(self.physics_parameters)})")
            raise ValueError("Mismatch between configurations and physics parameters")

        # Create feature matrix
        self._create_physics_feature_matrix()

        # Precompute optimization data for caching
        self._precompute_optimization_data()

    def _create_physics_feature_matrix(self):
        """Create and normalize feature matrix from physics parameters."""
        # ADD THIS CHECK
        if not self.physics_parameters:
            print("Warning: No physics parameters loaded, skipping feature matrix creation")
            self.feature_matrix = np.array([])
            self.feature_matrix_normalized = np.array([])
            return

        n_configs = len(self.physics_parameters)

                # Extract features - check if new parameter exists
        features = []
        has_edge_distance = 'avg_distance_to_edge' in self.physics_parameters[0] if self.physics_parameters else False

        # Determine which parameters to use
        if self.selected_parameters:
            # Use user-selected parameters
            params_to_use = self.selected_parameters
            self.feature_names = params_to_use.copy()
        else:
            # Use all available parameters
            if has_edge_distance:
                params_to_use = [
                    'avg_distance_from_core_center',
                    'min_inter_position_distance',
                    'clustering_coefficient',
                    'symmetry_balance',
                    'local_fuel_density',
                    'avg_distance_to_edge'
                ]
            else:
                params_to_use = [
                    'avg_distance_from_core_center',
                    'min_inter_position_distance',
                    'clustering_coefficient',
                    'symmetry_balance',
                    'local_fuel_density'
                ]
            self.feature_names = params_to_use.copy()

        # Extract selected features
        for i in range(n_configs):
            feature_vec = []
            for param_key in params_to_use:
                if param_key in self.physics_parameters[i]:
                    feature_vec.append(self.physics_parameters[i][param_key])
                else:
                    print(f"Warning: Parameter '{param_key}' not found in physics parameters")
                    feature_vec.append(0.0)  # Default value
            features.append(feature_vec)

        self.feature_matrix = np.array(features)
        self.feature_matrix_normalized = self.scaler.fit_transform(self.feature_matrix)

        # Fit the unified discretizer on the normalized features
        if not hasattr(self, '_discretizer'):
            # Initialize if not already done
            self._discretizer = UnifiedDiscretization(n_bins=15)

            # Fit on the full normalized feature matrix
            if self.feature_matrix_normalized is not None and len(self.feature_matrix_normalized) > 0:
                self._discretizer.fit(self.feature_matrix_normalized)

    def calculate_diversity_score_generic(self, selected_indices: List[int],
                                        distance_type: str = 'euclidean') -> float:
        """
        Calculate minimum pairwise distance using specified metric.
        This ensures consistency between selection and evaluation.

        Args:
            selected_indices: List of selected configuration indices
            distance_type: Type of distance metric ('euclidean', 'manhattan', 'jaccard')

        Returns:
            Minimum pairwise distance among selected configurations
        """
        if len(selected_indices) < 2:
            return 0.0

        min_distance = float('inf')
        n = len(selected_indices)

        for i in range(n):
            for j in range(i + 1, n):
                # Use the appropriate cached distance method
                if distance_type == 'euclidean':
                    dist = self.get_cached_euclidean_distance(selected_indices[i], selected_indices[j])
                elif distance_type == 'manhattan':
                    dist = self.get_cached_manhattan_distance(selected_indices[i], selected_indices[j])
                elif distance_type == 'jaccard':
                    dist = self.get_cached_jaccard_distance(selected_indices[i], selected_indices[j])
                else:
                    raise ValueError(f"Unknown distance type: {distance_type}")

                min_distance = min(min_distance, dist)

        return min_distance

    def calculate_diversity_score_lattice_generic(self, selected_indices: List[int],
                                                distance_type: str = 'euclidean') -> float:
        """
        Calculate minimum pairwise distance in lattice space using specified metric.

        Args:
            selected_indices: List of selected configuration indices
            distance_type: Type of distance metric ('euclidean', 'manhattan', 'jaccard')

        Returns:
            Minimum pairwise distance in lattice space
        """
        if len(selected_indices) < 2:
            return 0.0

        min_distance = float('inf')
        n = len(selected_indices)

        for i in range(n):
            for j in range(i + 1, n):
                # Use the appropriate lattice distance method
                if distance_type == 'euclidean':
                    dist = self.get_cached_lattice_euclidean_distance(selected_indices[i], selected_indices[j])
                elif distance_type == 'manhattan':
                    dist = self.get_cached_lattice_manhattan_distance(selected_indices[i], selected_indices[j])
                elif distance_type == 'jaccard':
                    # For lattice Jaccard, use symmetry-aware version
                    dist = self.calculate_symmetry_aware_jaccard_distance(selected_indices[i], selected_indices[j])
                else:
                    raise ValueError(f"Unknown distance type: {distance_type}")

                min_distance = min(min_distance, dist)

        return min_distance

    def find_closest_configuration(self, target_params: np.ndarray,
                                 used_indices: Set[int] = None) -> Tuple[int, float]:
        """Find the configuration whose parameters are closest to the target."""
        if used_indices is None:
            used_indices = set()

        # Calculate Euclidean distances to all configurations
        distances = np.linalg.norm(self.feature_matrix_normalized - target_params, axis=1)

        # Mask out already used configurations
        for idx in used_indices:
            distances[idx] = np.inf

        # Find the closest configuration
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]

        return closest_idx, min_distance

    def calculate_jaccard_distance(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard distance between two sets."""
        if len(set1) == 0 and len(set2) == 0:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union
        return 1 - jaccard_similarity

    def calculate_symmetry_aware_jaccard_distance(self, idx1: int, idx2: int) -> float:
        """Calculate Jaccard distance using pre-computed canonical forms."""
        # Use pre-computed canonical forms for efficient comparison
        if idx1 not in self.canonical_forms:
            self.canonical_forms[idx1] = get_canonical_form(self.irradiation_sets[idx1])
        if idx2 not in self.canonical_forms:
            self.canonical_forms[idx2] = get_canonical_form(self.irradiation_sets[idx2])

        # Compare canonical forms directly
        set1 = set(self.canonical_forms[idx1])
        set2 = set(self.canonical_forms[idx2])

        return self.calculate_jaccard_distance(set1, set2)

    def calculate_symmetry_aware_lattice_euclidean_distance(self, idx1: int, idx2: int) -> float:
        """Calculate lattice Euclidean distance using canonical forms."""
        # Get canonical forms
        if idx1 not in self.canonical_forms:
            self.canonical_forms[idx1] = get_canonical_form(self.irradiation_sets[idx1])
        if idx2 not in self.canonical_forms:
            self.canonical_forms[idx2] = get_canonical_form(self.irradiation_sets[idx2])

        # Convert canonical forms to coordinates
        coords1 = np.array([(i + 0.5, j + 0.5) for i, j in self.canonical_forms[idx1]])
        coords2 = np.array([(i + 0.5, j + 0.5) for i, j in self.canonical_forms[idx2]])

        # Calculate greedy Euclidean distance
        return self._calculate_greedy_euclidean_distance(coords1, coords2)

    def calculate_symmetry_aware_lattice_manhattan_distance(self, idx1: int, idx2: int) -> float:
        """Calculate lattice Manhattan distance using canonical forms."""
        # Get canonical forms
        if idx1 not in self.canonical_forms:
            self.canonical_forms[idx1] = get_canonical_form(self.irradiation_sets[idx1])
        if idx2 not in self.canonical_forms:
            self.canonical_forms[idx2] = get_canonical_form(self.irradiation_sets[idx2])

        # Convert canonical forms to coordinates
        coords1 = np.array([(i + 0.5, j + 0.5) for i, j in self.canonical_forms[idx1]])
        coords2 = np.array([(i + 0.5, j + 0.5) for i, j in self.canonical_forms[idx2]])

        # Calculate greedy Manhattan distance
        return self._calculate_greedy_manhattan_distance(coords1, coords2)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage for debugging and optimization monitoring."""
        stats = self.distance_cache.get_stats()
        stats.update({
            'coords_cache_size': len(self.coords_cache),
            'canonical_coords_cache_size': len(self.canonical_coords_cache),  # NEW
            'position_sets_computed': len(self.position_sets) if self.position_sets else 0,
            'discretized_features_cached': self.discretized_features_cache is not None
        })
        return stats

    def clear_caches(self):
        """Clear all caches to free memory if needed."""
        # Use the clear method of the efficient cache
        self.distance_cache.clear()

        # Clear coordinate caches
        self.coords_cache.clear()
        self.canonical_forms.clear()
        self.canonical_coords_cache.clear()  # NEW
        self.discretized_features_cache = None

    def sample(self, n_samples: int, **kwargs) -> Dict:
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement sample()")

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj

    def save_results(self, results: Dict):
        """Save sampling results to disk."""
        # Create output directory
        pkl_dir = SCRIPT_DIR / 'output/samples_picked/pkl'
        txt_dir = SCRIPT_DIR / 'output/samples_picked/txt'
        pkl_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        selected_indices = results['selected_indices']

        # Save pickle file
        pkl_data = {
            'method': self.method_name,
            'n_samples': len(selected_indices),
            'selected_indices': selected_indices,
            'configurations': [self.configurations[i] for i in selected_indices],
            'irradiation_sets': [self.irradiation_sets[i] for i in selected_indices],
        }
        pkl_data.update(results)

        pkl_path = pkl_dir / f'{self.method_name}_samples.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)

        # Save JSON file (without configurations)
        json_data = {
            'method': self.method_name,
            'n_samples': len(results['selected_indices']),
            'selected_indices': [int(idx) for idx in results['selected_indices']],
            'diversity_score': float(results['diversity_score']),
            'best_run': results.get('best_run'),
            'total_runs': results.get('total_runs')
        }

        # Add any distance metrics
        for key in ['min_jaccard_distance', 'min_euclidean_distance', 'min_manhattan_distance']:
            if key in results:
                json_data[key] = float(results[key])

        json_path = pkl_dir / f'{self.method_name}_samples.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Save text summary
        self._save_text_summary(results)

    def _save_text_summary(self, results: Dict):
        """Save text summary of results."""
        selected_indices = results['selected_indices']

        txt_path = SCRIPT_DIR / f'output/samples_picked/txt/{self.method_name}_summary.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{self.method_name.upper()} SAMPLING RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Number of samples: {len(selected_indices)}\n")

            if 'diversity_score' in results:
                f.write(f"Diversity score (min pairwise distance): {results['diversity_score']:.4f}\n")

            if 'matching_distances' in results:
                f.write(f"Average matching distance: {np.mean(results['matching_distances']):.4f}\n")

            if 'best_run' in results:
                f.write(f"Best run: #{results['best_run']} of {results['total_runs']}\n")

            # Add selection criterion information
            if 'algorithm' in results:
                f.write(f"Algorithm: {results['algorithm']}\n")
            if 'selection_metric' in results:
                f.write(f"Selection based on: {results['selection_metric']}\n")

            f.write("\nSelected configuration indices:\n")
            f.write(str(selected_indices))

            # Add the detailed configurations with grids
            f.write("\n\nDetailed Configurations:\n")
            f.write("="*60 + "\n")

            for i, idx in enumerate(selected_indices):
                f.write(f"\nConfiguration {i+1} (Index {idx}):\n")
                f.write(f"Irradiation positions: {self.irradiation_sets[idx]}\n")

                # Create and write the 8x8 grid
                grid = self._create_grid_from_positions(self.irradiation_sets[idx])
                f.write("Grid:\n")
                for row in grid:
                    f.write(' '.join(row) + '\n')

                # Write physics parameters
                f.write("\nPhysics parameters:\n")
                for param in self.feature_names:
                    f.write(f"  {param}: {self.physics_parameters[idx][param]:.4f}\n")
                f.write("-"*40 + "\n")

    def _create_grid_from_positions(self, positions):
        """Create an 8x8 grid representation from irradiation positions."""
        # Initialize 8x8 grid with default fuel configuration
        grid = [['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
                ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
                ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
                ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C']]

        # Place irradiation positions
        for i, (row, col) in enumerate(positions, 1):
            grid[row][col] = f'I_{i}'

        return grid

    def _discretize_features_unified(self, features):
        """Unified discretization method for all samplers."""
        if not hasattr(self, '_discretizer'):
            # Initialize if not already done
            self._discretizer = UnifiedDiscretization(n_bins=15)

            # Fit on the full normalized feature matrix
            if self.feature_matrix_normalized is not None and len(self.feature_matrix_normalized) > 0:
                self._discretizer.fit(self.feature_matrix_normalized)

        return self._discretizer.transform(features)
