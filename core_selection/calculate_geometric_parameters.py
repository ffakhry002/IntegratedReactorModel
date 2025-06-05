"""
Calculate geometric parameters for each core configuration according to the paper.
Saves results to the data folder.
"""

import numpy as np
import pickle
from typing import List, Tuple, Dict
import json
import os
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import argparse
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()


def load_configurations(filename='output/data/core_configurations_optimized.pkl'):
    """Load configurations from pickle file."""
    # If filename is relative, make it relative to SCRIPT_DIR
    if not Path(filename).is_absolute():
        filename = SCRIPT_DIR / filename

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['configurations'], data['irradiation_sets']


def calculate_average_distance_from_core_center(positions: List[Tuple[int, int]]) -> float:
    """
    Calculate average distance from core center.
    In an 8×8 grid, the geometric center is at coordinate (4, 4).
    Each grid square (i,j) has its center at (i+0.5, j+0.5).
    """
    core_center = (4, 4)

    distances = []
    for i, j in positions:
        # Center of grid square (i,j) is at (i+0.5, j+0.5)
        pos_center = (i + 0.5, j + 0.5)
        dx = pos_center[0] - core_center[0]
        dy = pos_center[1] - core_center[1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)

    return np.mean(distances)


def calculate_minimum_inter_position_distance(positions: List[Tuple[int, int]]) -> float:
    """
    Calculate the smallest pairwise distance between any two of the four selected positions.
    This affects neutron shadowing between positions.
    """
    min_distance = float('inf')
    n = len(positions)

    for i in range(n):
        for j in range(i + 1, n):
            # Distance between centers of grid squares
            pos1_center = (positions[i][0] + 0.5, positions[i][1] + 0.5)
            pos2_center = (positions[j][0] + 0.5, positions[j][1] + 0.5)

            dx = pos1_center[0] - pos2_center[0]
            dy = pos1_center[1] - pos2_center[1]
            dist = np.sqrt(dx**2 + dy**2)

            min_distance = min(min_distance, dist)

    return min_distance


def calculate_clustering_coefficient(positions: List[Tuple[int, int]]) -> float:
    """
    Calculate the radius of the smallest circle containing the centers of all four positions.
    This distinguishes between tightly grouped and dispersed configurations.
    """
    # Convert positions to centers
    centers = [(i + 0.5, j + 0.5) for i, j in positions]
    centers_array = np.array(centers)

    # Method: Find the circumcircle of the convex hull or use minimum enclosing circle
    # For simplicity, we'll use the maximum distance from centroid as an approximation
    # A more accurate method would use Welzl's algorithm for minimum enclosing circle

    centroid = np.mean(centers_array, axis=0)

    # Calculate distances from centroid to all points
    distances = []
    for center in centers_array:
        dist = np.linalg.norm(center - centroid)
        distances.append(dist)

    # The radius is the maximum distance from centroid
    # This is an upper bound on the minimum enclosing circle radius
    radius = max(distances)

    return radius


def calculate_symmetry_balance(positions: List[Tuple[int, int]]) -> float:
    """
    Calculate distance between the center of mass of the four position centers and the reactor center.
    The center of mass is calculated as the average of the four position centers.
    This captures whether configurations are balanced or skewed to one side.
    """
    reactor_center = (4, 4)

    # Calculate center of mass of the four positions
    centers = [(i + 0.5, j + 0.5) for i, j in positions]
    center_of_mass = np.mean(centers, axis=0)

    # Distance from center of mass to reactor center
    dx = center_of_mass[0] - reactor_center[0]
    dy = center_of_mass[1] - reactor_center[1]
    distance = np.sqrt(dx**2 + dy**2)

    return distance


def calculate_local_fuel_density(positions: List[Tuple[int, int]],
                                configuration: np.ndarray) -> float:
    """
    Calculate average number of fuel positions adjacent to each irradiation position.
    Adjacent means the 8 surrounding positions (including diagonals).
    Only counts actual fuel ('F'), not irradiation positions ('I') or coolant ('C').
    """
    fuel_counts = []

    # Define all 8 adjacent offsets (including diagonals)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for i, j in positions:
        fuel_count = 0

        for di, dj in offsets:
            ni, nj = i + di, j + dj

            # Check if position is within bounds
            if 0 <= ni < 8 and 0 <= nj < 8:
                # Count ONLY actual fuel positions ('F'), not irradiation ('I') or coolant ('C')
                if configuration[ni, nj] == 'F':
                    fuel_count += 1

        fuel_counts.append(fuel_count)

    # Return average fuel density
    return np.mean(fuel_counts)


def calculate_all_physics_parameters(configuration: np.ndarray,
                                   irradiation_positions: List[Tuple[int, int]]) -> Dict:
    """
    Calculate all five physics-informed parameters from the paper for a configuration.
    """
    params = {
        'irradiation_positions': irradiation_positions,
        'avg_distance_from_core_center': calculate_average_distance_from_core_center(irradiation_positions),
        'min_inter_position_distance': calculate_minimum_inter_position_distance(irradiation_positions),
        'clustering_coefficient': calculate_clustering_coefficient(irradiation_positions),
        'symmetry_balance': calculate_symmetry_balance(irradiation_positions),
        'local_fuel_density': calculate_local_fuel_density(irradiation_positions, configuration)
    }

    return params


def main():
    """Main function to calculate physics parameters for all configurations."""
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate geometric parameters for core configurations')
    parser.add_argument('--full', action='store_true',
                        help='Process full configuration set instead of symmetry-reduced')
    parser.add_argument('--restrict-6x6', action='store_true',
                        help='Process 6x6 restricted configurations')
    args = parser.parse_args()

    restriction_info = " (6x6 Central Square)" if args.restrict_6x6 else ""
    print(f"CORE CONFIGURATION PHYSICS PARAMETER CALCULATOR{restriction_info}")
    print("="*60)

    # Determine which configuration file to load
    suffix = "_6x6" if args.restrict_6x6 else ""

    if args.full:
        pkl_file = SCRIPT_DIR / f'output/data/all_configurations_before_symmetry{suffix}.pkl'
        output_file = SCRIPT_DIR / f'output/data/physics_parameters_full{suffix}.pkl'
        print(f"Processing FULL configuration set from {pkl_file.relative_to(SCRIPT_DIR)}")
    else:
        pkl_file = SCRIPT_DIR / f'output/data/core_configurations_optimized{suffix}.pkl'
        output_file = SCRIPT_DIR / f'output/data/physics_parameters{suffix}.pkl'
        print(f"Processing symmetry-reduced configuration set from {pkl_file.relative_to(SCRIPT_DIR)}")

    if args.restrict_6x6:
        print("NOTE: Processing 6x6 restricted configurations")
        print("Physics parameters are calculated considering the full 8x8 grid context")

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
        if args.restrict_6x6:
            print("  Make sure to use --restrict-6x6 flag when generating")
        return

    all_parameters = []
    last_percent = 0

    print("\nCalculating parameters...")
    print("-" * 60)

    for i, (config, irrad_pos) in enumerate(zip(configurations, irradiation_sets)):
        params = calculate_all_physics_parameters(config, irrad_pos)
        params['configuration_id'] = i
        all_parameters.append(params)

        # Progress update
        percent = int((i + 1) / len(configurations) * 100)
        if percent > last_percent:
            print(f"\rProgress: {percent}% | Processing configuration {i+1}/{len(configurations)}",
                  end='', flush=True)
            last_percent = percent

    print("\n\nSaving results...")

    # Save to data folder with appropriate suffix
    # Save JSON
    json_path = SCRIPT_DIR / f'output/data/physics_parameters{("_full" if args.full else "")}{suffix}.json'
    parameters_dict = {'parameters': all_parameters}
    with open(json_path, 'w') as f:
        json.dump(parameters_dict, f, indent=2, default=str)
    print(f"✓ Saved JSON to {json_path.relative_to(SCRIPT_DIR)}")

    # Save pickle
    with open(output_file, 'wb') as f:
        pickle.dump(parameters_dict, f)
    print(f"✓ Saved pickle to {output_file.relative_to(SCRIPT_DIR)}")

    # Create a summary in output/samples_picked/results
    results_dir = SCRIPT_DIR / 'output/samples_picked/results'
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_suffix = ("_full" if args.full else "") + suffix
    summary_path = results_dir / f'physics_parameters_summary{summary_suffix}.txt'

    with open(summary_path, 'w') as f:
        f.write(f"PHYSICS-INFORMED PARAMETERS SUMMARY ({'FULL SET' if args.full else 'REDUCED SET'})\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total configurations analyzed: {len(all_parameters)}\n\n")

        # Parameter names matching the paper
        param_names = {
            'avg_distance_from_core_center': 'Average Distance from Core Center',
            'min_inter_position_distance': 'Minimum Inter-Position Distance',
            'clustering_coefficient': 'Clustering Coefficient (Radius)',
            'symmetry_balance': 'Symmetry Balance',
            'local_fuel_density': 'Local Fuel Density'
        }

        f.write("PARAMETER STATISTICS:\n")
        f.write("-"*60 + "\n\n")

        for param_key, param_name in param_names.items():
            values = [p[param_key] for p in all_parameters]
            f.write(f"{param_name}:\n")
            f.write(f"  Min:    {min(values):.4f}\n")
            f.write(f"  Max:    {max(values):.4f}\n")
            f.write(f"  Mean:   {np.mean(values):.4f}\n")
            f.write(f"  Median: {np.median(values):.4f}\n")
            f.write(f"  Std:    {np.std(values):.4f}\n\n")

        # Show parameter ranges for LHS/Sobol sampling
        f.write("\nPARAMETER RANGES FOR CONTINUOUS SAMPLING:\n")
        f.write("-"*60 + "\n")
        for param_key, param_name in param_names.items():
            values = [p[param_key] for p in all_parameters]
            f.write(f"{param_name}: [{min(values):.4f}, {max(values):.4f}]\n")

    print(f"✓ Saved summary to {summary_path.relative_to(SCRIPT_DIR)}")

    print("\n" + "="*60)
    print("PARAMETER CALCULATION COMPLETE!")
    print("="*60)
    print("\nSaved files:")
    print(f"  {json_path.relative_to(SCRIPT_DIR)}")
    print(f"  {output_file.relative_to(SCRIPT_DIR)}")
    print(f"  {summary_path.relative_to(SCRIPT_DIR)}")

    if not args.full:
        print("\nTo calculate parameters for the FULL configuration set, run:")
        print("  python calculate_geometric_parameters.py --full")


if __name__ == "__main__":
    main()
