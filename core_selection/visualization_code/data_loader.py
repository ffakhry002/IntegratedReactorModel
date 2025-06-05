"""
Data loading utilities for sampling visualization.
Handles loading configurations, physics parameters, and sample results.
FIXED: Smart loading based on actual indices in the results
UPDATED: Support for 6x6 restricted configurations
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Get the script directory (parent of visualization_code)
SCRIPT_DIR = Path(__file__).parent.parent.absolute()


def detect_6x6_mode():
    """Detect if we're working with 6x6 restricted data by checking for suffixed files."""
    # Check if 6x6 configuration files exist
    if (SCRIPT_DIR / 'output/data/core_configurations_optimized_6x6.pkl').exists():
        return True
    return False


def load_core_configurations(full_set=False, use_6x6=None):
    """Load core configurations from pickle file."""
    # Auto-detect 6x6 mode if not specified
    if use_6x6 is None:
        use_6x6 = detect_6x6_mode()

    suffix = "_6x6" if use_6x6 else ""

    if full_set:
        # Load all configurations before symmetry reduction
        file_path = SCRIPT_DIR / f'output/data/all_configurations_before_symmetry{suffix}.pkl'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # Load symmetry-reduced configurations
        file_path = SCRIPT_DIR / f'output/data/core_configurations_optimized{suffix}.pkl'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    return data['configurations'], data['irradiation_sets']


def load_configurations_for_method(method_name, max_index=None, use_6x6=None):
    """Load appropriate configuration set based on method type and maximum index.
    FIXED: Properly determines which dataset to load based on actual indices."""

    # Auto-detect 6x6 mode if not specified
    if use_6x6 is None:
        use_6x6 = detect_6x6_mode()

    geometric_methods = ['random_geometric', 'halton', 'sobol', 'lhs',
                        'jaccard_geometric', 'euclidean_geometric', 'manhattan_geometric',
                        'jaccard_geometric_kmedoids', 'euclidean_geometric_kmedoids',
                        'manhattan_geometric_kmedoids']

    # FIXED: If we have a max_index >= 34119, we MUST load the full set
    # For 6x6 mode, the threshold would be different (need to check actual numbers)
    threshold = 34119 if not use_6x6 else 9999  # Adjust based on actual 6x6 reduced set size

    if max_index is not None and max_index >= threshold:
        print(f"  Max index {max_index} requires FULL configuration set")
        try:
            configs, irrad_sets = load_core_configurations(full_set=True, use_6x6=use_6x6)
            config_count = 270725 if not use_6x6 else "TBD"  # Update with actual 6x6 full count
            print(f"  Loaded FULL configuration set ({len(configs):,} configs) for {method_name}")
            return configs, irrad_sets, 'full'
        except FileNotFoundError:
            raise ValueError(f"Method {method_name} has indices up to {max_index} but full configuration set not found!")

    # For geometric methods without high indices, try full set first
    if method_name in geometric_methods:
        try:
            configs, irrad_sets = load_core_configurations(full_set=True, use_6x6=use_6x6)
            print(f"  Loaded FULL configuration set ({len(configs):,} configs) for {method_name}")
            return configs, irrad_sets, 'full'
        except FileNotFoundError:
            print(f"  Warning: Full configuration set not found, using reduced set for {method_name}")
            configs, irrad_sets = load_core_configurations(full_set=False, use_6x6=use_6x6)
            print(f"  Loaded symmetry-reduced set ({len(configs):,} configs) for {method_name}")
            return configs, irrad_sets, 'reduced'
    else:
        # Lattice-based methods always use reduced set
        configs, irrad_sets = load_core_configurations(full_set=False, use_6x6=use_6x6)
        print(f"  Loaded symmetry-reduced set ({len(configs):,} configs) for {method_name}")
        return configs, irrad_sets, 'reduced'


def load_physics_parameters(filename=None, use_6x6=None):
    """Load physics parameters from pickle file."""
    # Auto-detect 6x6 mode if not specified
    if use_6x6 is None:
        use_6x6 = detect_6x6_mode()

    if filename is None:
        suffix = "_6x6" if use_6x6 else ""
        filename = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'
    elif not Path(filename).is_absolute():
        filename = SCRIPT_DIR / filename

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['parameters']


def load_physics_parameters_for_method(method_name, max_index=None, use_6x6=None):
    """Load appropriate physics parameters based on method type.
    FIXED: Properly determines which dataset to load based on actual indices."""

    # Auto-detect 6x6 mode if not specified
    if use_6x6 is None:
        use_6x6 = detect_6x6_mode()

    suffix = "_6x6" if use_6x6 else ""

    geometric_methods = ['random_geometric', 'halton', 'sobol', 'lhs',
                        'jaccard_geometric', 'euclidean_geometric', 'manhattan_geometric',
                        'jaccard_geometric_kmedoids', 'euclidean_geometric_kmedoids',
                        'manhattan_geometric_kmedoids']

    # FIXED: If we have a max_index >= 34119, we MUST load the full set
    threshold = 34119 if not use_6x6 else 9999  # Adjust based on actual 6x6 reduced set size

    if max_index is not None and max_index >= threshold:
        print(f"  Max index {max_index} requires FULL physics parameters")
        full_file = SCRIPT_DIR / f'output/data/physics_parameters_full{suffix}.pkl'
        if full_file.exists():
            with open(full_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded physics parameters for FULL set ({len(data['parameters']):,} params)")
            return data['parameters'], 'full'
        else:
            raise ValueError(f"Method {method_name} has indices up to {max_index} but full physics parameters not found!")

    # For geometric methods without high indices, prefer full set
    if method_name in geometric_methods:
        full_file = SCRIPT_DIR / f'output/data/physics_parameters_full{suffix}.pkl'
        if full_file.exists():
            with open(full_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded physics parameters for FULL set ({len(data['parameters']):,} params)")
            return data['parameters'], 'full'
        else:
            print(f"  Warning: Full physics parameters not found, using reduced set")
            reduced_file = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'
            with open(reduced_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded physics parameters for reduced set ({len(data['parameters']):,} params)")
            return data['parameters'], 'reduced'
    else:
        # Lattice-based methods always use reduced set
        reduced_file = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'
        print(f"  Loaded physics parameters for reduced set")
        with open(reduced_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded physics parameters for reduced set ({len(data['parameters']):,} params)")
        return data['parameters'], 'reduced'


def load_sampling_results(method_name):
    """Load sampling results for a specific method."""
    pkl_file = SCRIPT_DIR / f'output/samples_picked/pkl/{method_name}_samples.pkl'
    json_file = SCRIPT_DIR / f'output/samples_picked/pkl/{method_name}_samples.json'

    # Try loading pickle file first
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

    # Fallback to JSON file
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)

    return None


def get_max_index_from_results(sample_data):
    """Get the maximum index from sampling results.
    NEW FUNCTION to help determine which dataset to load."""
    if sample_data and 'selected_indices' in sample_data:
        indices = sample_data['selected_indices']
        if indices:
            return max(indices)
    return None


def load_data_for_visualization(method_name):
    """Load all necessary data for visualizing a method's results.
    NEW FUNCTION that handles the complete loading process."""

    # First, load the sampling results to get the indices
    sample_data = load_sampling_results(method_name)
    if sample_data is None:
        raise ValueError(f"No results found for method {method_name}")

    # Get the maximum index to determine which dataset we need
    max_index = get_max_index_from_results(sample_data)

    # Load configurations and physics parameters based on max index
    configs, irrad_sets, config_type = load_configurations_for_method(method_name, max_index)
    physics_params, param_type = load_physics_parameters_for_method(method_name, max_index)

    # Verify consistency
    if len(configs) != len(physics_params):
        raise ValueError(f"Mismatch: {len(configs)} configs vs {len(physics_params)} physics params")

    # Verify indices are valid
    if max_index is not None and max_index >= len(configs):
        raise ValueError(f"Invalid index {max_index} for dataset with {len(configs)} configurations")

    return {
        'sample_data': sample_data,
        'configurations': configs,
        'irradiation_sets': irrad_sets,
        'physics_params': physics_params,
        'config_type': config_type,
        'param_type': param_type,
        'max_index': max_index
    }


def get_all_methods():
    """Get list of all methods that have results."""
    methods = []
    pkl_dir = SCRIPT_DIR / 'output/samples_picked/pkl'

    if pkl_dir.exists():
        for file in pkl_dir.iterdir():
            if file.suffix == '.pkl' and file.name.endswith('_samples.pkl'):
                method = file.stem.replace('_samples', '')
                methods.append(method)

    return sorted(methods)


def load_method_indices(method):
    """Load selected indices for a specific method."""
    pkl_file = SCRIPT_DIR / f'output/samples_picked/pkl/{method}_samples.pkl'

    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            return data.get('selected_indices', [])

    return []


def get_method_colors() -> Dict[str, str]:
    """Get color scheme for all methods."""
    lattice_methods = [
        'lhs_lattice', 'sobol_lattice', 'halton_lattice',
        'jaccard_lattice', 'euclidean_lattice', 'manhattan_lattice', 'random_lattice',
        'euclidean_lattice_geometric_diversity',
        'jaccard_lattice_kmedoids', 'euclidean_lattice_kmedoids', 'manhattan_lattice_kmedoids'
    ]

    geometric_methods = [
        'lhs', 'sobol', 'halton',
        'jaccard_geometric', 'euclidean_geometric', 'manhattan_geometric', 'random_geometric',
        'jaccard_geometric_kmedoids', 'euclidean_geometric_kmedoids', 'manhattan_geometric_kmedoids'
    ]

    # Color schemes organized by type - extend for k-medoids
    lattice_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FFB6C1',
                      '#98D8C8',  # Added color for euclidean_lattice_geometric_diversity
                      '#8B4513', '#2F4F4F', '#483D8B']  # Added darker colors for k-medoids
    geometric_colors = ['#FD79A8', '#6C5CE7', '#A29BFE', '#74B9FF', '#00B894', '#FDCB6E', '#FF8C69',
                        '#8B008B', '#006400', '#B22222']  # Added darker colors for k-medoids

    method_colors = dict(zip(lattice_methods, lattice_colors))
    method_colors.update(dict(zip(geometric_methods, geometric_colors)))

    return method_colors


def get_method_lists() -> Tuple[List[str], List[str]]:
    """Get lists of lattice and geometric methods."""
    lattice_methods = [
        'lhs_lattice', 'sobol_lattice', 'halton_lattice',
        'jaccard_lattice', 'euclidean_lattice', 'manhattan_lattice', 'random_lattice',
        'euclidean_lattice_geometric_diversity',
        'jaccard_lattice_kmedoids', 'euclidean_lattice_kmedoids', 'manhattan_lattice_kmedoids'
    ]

    geometric_methods = [
        'lhs', 'sobol', 'halton',
        'jaccard_geometric', 'euclidean_geometric', 'manhattan_geometric', 'random_geometric',
        'jaccard_geometric_kmedoids', 'euclidean_geometric_kmedoids', 'manhattan_geometric_kmedoids'
    ]

    return lattice_methods, geometric_methods


def create_summary_statistics_data(samples_data, lattice_methods, geometric_methods):
    """Calculate summary statistics for export."""
    import numpy as np

    # Calculate statistics for each method type
    lattice_diversities = [samples_data[m]['diversity_score'] for m in lattice_methods if m in samples_data]
    geometric_diversities = [samples_data[m]['diversity_score'] for m in geometric_methods if m in samples_data]

    summary = {
        'Lattice Methods': {
            'count': len(lattice_diversities),
            'mean_diversity': np.mean(lattice_diversities) if lattice_diversities else 0,
            'std_diversity': np.std(lattice_diversities) if lattice_diversities else 0,
            'min_diversity': np.min(lattice_diversities) if lattice_diversities else 0,
            'max_diversity': np.max(lattice_diversities) if lattice_diversities else 0
        },
        'Geometric Methods': {
            'count': len(geometric_diversities),
            'mean_diversity': np.mean(geometric_diversities) if geometric_diversities else 0,
            'std_diversity': np.std(geometric_diversities) if geometric_diversities else 0,
            'min_diversity': np.min(geometric_diversities) if geometric_diversities else 0,
            'max_diversity': np.max(geometric_diversities) if geometric_diversities else 0
        }
    }

    return summary


def load_sample_data(method_name: str) -> Optional[Dict]:
    """Alias for load_sampling_results for backward compatibility."""
    return load_sampling_results(method_name)


def load_all_results(methods: List[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Load all available sampling results.

    Args:
        methods: List of method names to check

    Returns:
        Tuple of (results_data, samples_data) dictionaries
    """
    results_data = {}
    samples_data = {}

    for method in methods:
        sample_data = load_sampling_results(method)
        if sample_data is not None:
            samples_data[method] = sample_data
            # Also store in results_data for compatibility
            results_data[method] = sample_data

    return results_data, samples_data
