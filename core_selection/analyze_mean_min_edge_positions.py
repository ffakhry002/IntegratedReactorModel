# !/usr/bin/env python3
"""
Diagnostic script to analyze edge positions in samples_picked_mean and samples_picked_min folders.
Reports statistics about irradiation positions on edges or adjacent to coolant.
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()


def is_edge_position(i, j, configuration):
    """Check if a position is on the edge of the core.

    Edge means:
    1. On the outer edge of the 8x8 grid (i=0, i=7, j=0, or j=7)
    2. Adjacent to a coolant position 'C'

    Parameters
    ----------
    i : int
        Row position
    j : int
        Column position
    configuration : np.ndarray
        2D array representing the reactor core layout

    Returns
    -------
    bool
        True if position is on edge, False otherwise
    """
    # Check if on outer edge
    if i == 0 or i == 7 or j == 0 or j == 7:
        return True

    # Check if adjacent to coolant (including diagonals)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for di, dj in offsets:
        ni, nj = i + di, j + dj
        if 0 <= ni < 8 and 0 <= nj < 8:
            if configuration[ni, nj] == 'C':
                return True

    return False


def analyze_edge_positions(configurations, irradiation_sets):
    """Analyze edge positions across all configurations.

    Parameters
    ----------
    configurations : list
        List of 2D arrays representing reactor core layouts
    irradiation_sets : list
        List of irradiation position sets for each configuration

    Returns
    -------
    dict
        Dictionary containing statistics about edge positions including:
        - total_positions: Total number of irradiation positions
        - edge_positions: Number of positions on edges
        - cores_by_edge_count: Distribution of cores by edge count
        - edge_details: Detailed classification of edge types
        - core_edge_counts: List of edge counts for each core
    """
    total_positions = 0
    edge_positions = 0

    # Count cores by number of edge positions
    cores_by_edge_count = defaultdict(int)

    # Detailed edge position info
    edge_details = {
        'outer_edge': 0,
        'coolant_adjacent': 0,
        'both': 0  # positions that are both on outer edge AND adjacent to coolant
    }

    # Store individual core edge counts for detailed analysis
    core_edge_counts = []

    for config_idx, (config, irrad_pos) in enumerate(zip(configurations, irradiation_sets)):
        edge_count_for_core = 0

        for i, j in irrad_pos:
            total_positions += 1

            # Check if on edge
            if is_edge_position(i, j, config):
                edge_positions += 1
                edge_count_for_core += 1

                # Detailed classification
                on_outer = (i == 0 or i == 7 or j == 0 or j == 7)
                adjacent_coolant = False

                # Check adjacency to coolant
                offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 8 and 0 <= nj < 8 and config[ni, nj] == 'C':
                        adjacent_coolant = True
                        break

                if on_outer and adjacent_coolant:
                    edge_details['both'] += 1
                elif on_outer:
                    edge_details['outer_edge'] += 1
                elif adjacent_coolant:
                    edge_details['coolant_adjacent'] += 1

        # Count this core by its number of edge positions
        cores_by_edge_count[edge_count_for_core] += 1
        core_edge_counts.append(edge_count_for_core)

    return {
        'total_positions': total_positions,
        'edge_positions': edge_positions,
        'cores_by_edge_count': dict(cores_by_edge_count),
        'edge_details': edge_details,
        'core_edge_counts': core_edge_counts
    }


def analyze_folder_samples(folder_name, configurations, irradiation_sets):
    """Analyze edge positions in sampled cores from a specific folder.

    Parameters
    ----------
    folder_name : str
        Name of the folder containing sampling results
    configurations : list
        List of 2D arrays representing reactor core layouts
    irradiation_sets : list
        List of irradiation position sets for each configuration

    Returns
    -------
    dict or None
        Dictionary containing analysis results for each method, or None if no results found
    """
    samples_dir = SCRIPT_DIR / f'output/{folder_name}/pkl'
    if not samples_dir.exists():
        print(f"No sampling results found in {folder_name}")
        return None

    print(f"\n{'='*60}")
    print(f"ANALYZING {folder_name.upper()}")
    print('='*60)

    # Get all sampling result files
    sample_files = list(samples_dir.glob('*_samples.pkl'))

    all_results = {}

    # Define geometric methods that might use full dataset
    geometric_methods = ['random_geometric', 'halton', 'sobol', 'lhs',
                        'jaccard_geometric', 'euclidean_geometric', 'manhattan_geometric',
                        'jaccard_geometric_kmedoids', 'euclidean_geometric_kmedoids',
                        'manhattan_geometric_kmedoids']

    for sample_file in sorted(sample_files):
        method_name = sample_file.stem.replace('_samples', '')

        # Load sampling results
        with open(sample_file, 'rb') as f:
            sample_data = pickle.load(f)

        if 'selected_indices' not in sample_data:
            continue

        indices = sample_data['selected_indices']

        # Check if we need to load full dataset for this method
        max_index = max(indices) if indices else 0
        configs_to_use = configurations
        irrad_to_use = irradiation_sets

        # If indices exceed current configuration set size, load full set
        if max_index >= len(configurations) and method_name in geometric_methods:
            print(f"  Loading full dataset for {method_name} (max index: {max_index})")
            try:
                full_pkl = SCRIPT_DIR / 'output/data/all_configurations_before_symmetry.pkl'
                with open(full_pkl, 'rb') as f:
                    full_data = pickle.load(f)
                configs_to_use = full_data['configurations']
                irrad_to_use = full_data['irradiation_sets']
            except FileNotFoundError:
                print(f"  WARNING: Could not load full dataset for {method_name}, skipping...")
                continue

        # Filter configurations and irradiation sets to only sampled ones
        sampled_configs = [configs_to_use[i] for i in indices]
        sampled_irrad = [irrad_to_use[i] for i in indices]

        # Analyze these specific samples
        results = analyze_edge_positions(sampled_configs, sampled_irrad)
        results['indices'] = indices  # Store indices for reference
        all_results[method_name] = results

        # Print results for this method
        print(f"\n{method_name}:")
        print("-" * len(method_name))
        print(f"  Sampled cores: {len(indices)}")
        print(f"  Total positions: {results['total_positions']}")
        print(f"  Edge positions: {results['edge_positions']} ({results['edge_positions']/results['total_positions']*100:.1f}%)")
        print(f"  Average edge positions per core: {results['edge_positions']/len(indices):.2f}")

        # Edge position statistics
        edge_counts = results['core_edge_counts']
        print(f"  Edge position statistics:")
        print(f"    Min: {min(edge_counts)}")
        print(f"    Max: {max(edge_counts)}")
        print(f"    Mean: {np.mean(edge_counts):.2f}")
        print(f"    Std: {np.std(edge_counts):.2f}")

        # Distribution
        print(f"  Distribution:")
        for edge_count in range(5):
            count = results['cores_by_edge_count'].get(edge_count, 0)
            if count > 0:
                percentage = count / len(indices) * 100
                print(f"    {edge_count} edges: {count} cores ({percentage:.1f}%)")

    return all_results


def compare_mean_min_results(mean_results, min_results):
    """Compare results between mean and min folders.

    Parameters
    ----------
    mean_results : dict
        Analysis results from mean folder
    min_results : dict
        Analysis results from min folder

    Returns
    -------
    None
    """
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN MEAN AND MIN SAMPLES")
    print('='*60)

    # Find common methods
    common_methods = set(mean_results.keys()) & set(min_results.keys())

    for method in sorted(common_methods):
        print(f"\n{method}:")
        print("-" * len(method))

        mean_data = mean_results[method]
        min_data = min_results[method]

        # Calculate percentages
        mean_percent = mean_data['edge_positions'] / mean_data['total_positions'] * 100
        min_percent = min_data['edge_positions'] / min_data['total_positions'] * 100

        # Average per core
        mean_avg = mean_data['edge_positions'] / len(mean_data['indices'])
        min_avg = min_data['edge_positions'] / len(min_data['indices'])

        print(f"  Edge positions percentage:")
        print(f"    Mean folder: {mean_percent:.1f}%")
        print(f"    Min folder:  {min_percent:.1f}%")
        print(f"    Difference:  {mean_percent - min_percent:+.1f}%")

        print(f"  Average edge positions per core:")
        print(f"    Mean folder: {mean_avg:.2f}")
        print(f"    Min folder:  {min_avg:.2f}")
        print(f"    Difference:  {mean_avg - min_avg:+.2f}")

        # Compare distributions
        print(f"  Distribution comparison:")
        for edge_count in range(5):
            mean_count = mean_data['cores_by_edge_count'].get(edge_count, 0)
            min_count = min_data['cores_by_edge_count'].get(edge_count, 0)

            mean_pct = mean_count / len(mean_data['indices']) * 100 if mean_count > 0 else 0
            min_pct = min_count / len(min_data['indices']) * 100 if min_count > 0 else 0

            if mean_count > 0 or min_count > 0:
                print(f"    {edge_count} edges: Mean={mean_pct:.1f}%, Min={min_pct:.1f}% (diff={mean_pct-min_pct:+.1f}%)")


def main():
    """Main function to run the edge position diagnostic.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description='Analyze edge positions in mean/min sample folders')
    parser.add_argument('--full', action='store_true',
                        help='Use full configuration set for comparison')
    args = parser.parse_args()

    print("EDGE POSITION DIAGNOSTIC FOR MEAN/MIN SAMPLES")
    print("="*60)

    # Load base configurations for comparison
    if args.full:
        pkl_file = SCRIPT_DIR / 'output/data/all_configurations_before_symmetry.pkl'
        print(f"Using FULL configuration set as base")
    else:
        pkl_file = SCRIPT_DIR / 'output/data/core_configurations_optimized.pkl'
        print(f"Using symmetry-reduced configuration set as base")

    # Load configurations
    print("\nLoading configurations...")
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        configurations = data['configurations']
        irradiation_sets = data['irradiation_sets']
        print(f"✓ Loaded {len(configurations):,} configurations")
    except FileNotFoundError:
        print(f"✗ Error: Could not find {pkl_file.relative_to(SCRIPT_DIR)}")
        print("  Please run generate_core_configurations.py first")
        return

    # Analyze samples_picked_mean
    mean_results = analyze_folder_samples('samples_picked_mean', configurations, irradiation_sets)

    # Analyze samples_picked_min
    min_results = analyze_folder_samples('samples_picked_min', configurations, irradiation_sets)

    # Compare results if both folders had data
    if mean_results and min_results:
        compare_mean_min_results(mean_results, min_results)

    # Save detailed results
    output_file = SCRIPT_DIR / 'output/edge_position_diagnostic_mean_min.txt'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("EDGE POSITION DIAGNOSTIC - MEAN VS MIN SAMPLES\n")
        f.write("="*60 + "\n\n")

        # Write mean folder results
        if mean_results:
            f.write("SAMPLES_PICKED_MEAN RESULTS\n")
            f.write("="*60 + "\n")

            for method_name, method_results in sorted(mean_results.items()):
                f.write(f"\n{method_name}:\n")
                f.write("-" * len(method_name) + "\n")

                n_samples = len(method_results['indices'])
                f.write(f"  Sampled cores: {n_samples}\n")
                f.write(f"  Total positions: {method_results['total_positions']}\n")
                f.write(f"  Edge positions: {method_results['edge_positions']} ({method_results['edge_positions']/method_results['total_positions']*100:.1f}%)\n")
                f.write(f"  Average edge positions per core: {method_results['edge_positions']/n_samples:.2f}\n\n")

                # Edge statistics
                edge_counts = method_results['core_edge_counts']
                f.write(f"  Edge position statistics:\n")
                f.write(f"    Min: {min(edge_counts)}\n")
                f.write(f"    Max: {max(edge_counts)}\n")
                f.write(f"    Mean: {np.mean(edge_counts):.2f}\n")
                f.write(f"    Std: {np.std(edge_counts):.2f}\n\n")

                f.write("  Distribution:\n")
                for edge_count in range(5):
                    count = method_results['cores_by_edge_count'].get(edge_count, 0)
                    if count > 0:
                        percentage = count / n_samples * 100
                        f.write(f"    {edge_count} edges: {count} cores ({percentage:.1f}%)\n")

        # Write min folder results
        if min_results:
            f.write("\n\nSAMPLES_PICKED_MIN RESULTS\n")
            f.write("="*60 + "\n")

            for method_name, method_results in sorted(min_results.items()):
                f.write(f"\n{method_name}:\n")
                f.write("-" * len(method_name) + "\n")

                n_samples = len(method_results['indices'])
                f.write(f"  Sampled cores: {n_samples}\n")
                f.write(f"  Total positions: {method_results['total_positions']}\n")
                f.write(f"  Edge positions: {method_results['edge_positions']} ({method_results['edge_positions']/method_results['total_positions']*100:.1f}%)\n")
                f.write(f"  Average edge positions per core: {method_results['edge_positions']/n_samples:.2f}\n\n")

                # Edge statistics
                edge_counts = method_results['core_edge_counts']
                f.write(f"  Edge position statistics:\n")
                f.write(f"    Min: {min(edge_counts)}\n")
                f.write(f"    Max: {max(edge_counts)}\n")
                f.write(f"    Mean: {np.mean(edge_counts):.2f}\n")
                f.write(f"    Std: {np.std(edge_counts):.2f}\n\n")

                f.write("  Distribution:\n")
                for edge_count in range(5):
                    count = method_results['cores_by_edge_count'].get(edge_count, 0)
                    if count > 0:
                        percentage = count / n_samples * 100
                        f.write(f"    {edge_count} edges: {count} cores ({percentage:.1f}%)\n")

        # Write comparison
        if mean_results and min_results:
            f.write("\n\nCOMPARISON BETWEEN MEAN AND MIN\n")
            f.write("="*60 + "\n")

            common_methods = set(mean_results.keys()) & set(min_results.keys())

            for method in sorted(common_methods):
                f.write(f"\n{method}:\n")
                f.write("-" * len(method) + "\n")

                mean_data = mean_results[method]
                min_data = min_results[method]

                # Calculate percentages
                mean_percent = mean_data['edge_positions'] / mean_data['total_positions'] * 100
                min_percent = min_data['edge_positions'] / min_data['total_positions'] * 100

                # Average per core
                mean_avg = mean_data['edge_positions'] / len(mean_data['indices'])
                min_avg = min_data['edge_positions'] / len(min_data['indices'])

                f.write(f"  Edge positions percentage:\n")
                f.write(f"    Mean folder: {mean_percent:.1f}%\n")
                f.write(f"    Min folder:  {min_percent:.1f}%\n")
                f.write(f"    Difference:  {mean_percent - min_percent:+.1f}%\n\n")

                f.write(f"  Average edge positions per core:\n")
                f.write(f"    Mean folder: {mean_avg:.2f}\n")
                f.write(f"    Min folder:  {min_avg:.2f}\n")
                f.write(f"    Difference:  {mean_avg - min_avg:+.2f}\n")

    print(f"\n✓ Detailed results saved to: {output_file.relative_to(SCRIPT_DIR)}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
