#!/usr/bin/env python3
"""
Parametric Simulation Results Analyzer
Analyzes all results.txt files to extract k-effective and flux values with their uncertainties
"""

import os
import re
import numpy as np
from pathlib import Path
import glob

def parse_results_file(filepath):
    """Parse a single results.txt file and extract all relevant data"""
    data = {
        'keff': None,
        'keff_std': None,
        'positions': {}
    }

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract K-effective
        keff_pattern = r'K-effective:\s+([0-9.]+)\s+±\s+([0-9.]+)'
        keff_match = re.search(keff_pattern, content)
        if keff_match:
            data['keff'] = float(keff_match.group(1))
            data['keff_std'] = float(keff_match.group(2))

        # Extract position data
        # Pattern to find position blocks
        position_blocks = re.findall(r'Position (I_\d+):\s*\n'
                                   r'Total flux:\s+([0-9.e+\-]+)\s+±\s+([0-9.e+\-]+).*?\n'
                                   r'Thermal\s+([0-9.e+\-]+)\s+([0-9.e+\-]+)\s*\n'
                                   r'Epithermal\s+([0-9.e+\-]+)\s+([0-9.e+\-]+)\s*\n'
                                   r'Fast\s+([0-9.e+\-]+)\s+([0-9.e+\-]+)', content)

        for match in position_blocks:
            pos_name = match[0]
            data['positions'][pos_name] = {
                'total_flux': float(match[1]),
                'total_flux_std': float(match[2]),
                'thermal': float(match[3]),
                'thermal_std': float(match[4]),
                'epithermal': float(match[5]),
                'epithermal_std': float(match[6]),
                'fast': float(match[7]),
                'fast_std': float(match[8])
            }

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

    return data

def calculate_percent_uncertainty(value, std_dev):
    """Calculate percentage uncertainty"""
    if value == 0:
        return 0
    return (std_dev / value) * 100

def analyze_parametric_results(base_dir):
    """Analyze all results.txt files in the parametric simulation directory"""

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist!")
        return

    # Find all results.txt files
    results_files = []
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))

    for run_dir in sorted(run_dirs):
        results_file = os.path.join(run_dir, "results.txt")
        if os.path.exists(results_file):
            results_files.append(results_file)

    print(f"Found {len(results_files)} results.txt files")

    # Parse all files
    all_data = []
    for results_file in results_files:
        data = parse_results_file(results_file)
        if data and data['keff'] is not None:
            all_data.append(data)

    print(f"Successfully parsed {len(all_data)} files")

    if not all_data:
        print("No valid data found!")
        return

    # Collect statistics
    stats = {}

    # K-effective data collection
    keff_values = []
    keff_stds = []
    keff_percentages = []

    for data in all_data:
        if data['keff'] and data['keff_std']:
            keff_values.append(data['keff'])
            keff_stds.append(data['keff_std'])
            pct = calculate_percent_uncertainty(data['keff'], data['keff_std'])
            keff_percentages.append(pct)

    # K-effective statistics
    if keff_values:
        stats['keff_values'] = {
            'min': np.min(keff_values),
            'max': np.max(keff_values),
            'mean': np.mean(keff_values),
            'std': np.std(keff_values),
            'count': len(keff_values)
        }

        stats['keff_stds'] = {
            'min': np.min(keff_stds),
            'max': np.max(keff_stds),
            'mean': np.mean(keff_stds),
            'std': np.std(keff_stds),
            'count': len(keff_stds)
        }

        stats['keff_uncertainty_pct'] = {
            'min': np.min(keff_percentages),
            'max': np.max(keff_percentages),
            'mean': np.mean(keff_percentages),
            'std': np.std(keff_percentages),
            'count': len(keff_percentages)
        }

    # For each flux type, collect values, std devs, and percentages from ALL positions
    flux_types = ['total_flux', 'thermal', 'epithermal', 'fast']

    for flux_type in flux_types:
        values = []
        stds = []
        percentages = []

        # Collect from all positions across all data files
        for data in all_data:
            for pos_name, pos_data in data['positions'].items():
                if flux_type in pos_data and f'{flux_type}_std' in pos_data:
                    value = pos_data[flux_type]
                    std_dev = pos_data[f'{flux_type}_std']
                    values.append(value)
                    stds.append(std_dev)
                    pct = calculate_percent_uncertainty(value, std_dev)
                    percentages.append(pct)

        # Flux value statistics
        if values:
            stats[f'{flux_type}_values'] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }

            stats[f'{flux_type}_stds'] = {
                'min': np.min(stds),
                'max': np.max(stds),
                'mean': np.mean(stds),
                'std': np.std(stds),
                'count': len(stds)
            }

            stats[f'{flux_type}_uncertainty_pct'] = {
                'min': np.min(percentages),
                'max': np.max(percentages),
                'mean': np.mean(percentages),
                'std': np.std(percentages),
                'count': len(percentages)
            }

    # Write results to file
    output_file = os.path.join(base_dir, "uncertainty_analysis_results.txt")

    with open(output_file, 'w') as f:
        f.write("PARAMETRIC SIMULATION UNCERTAINTY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis of {len(all_data)} simulation runs\n")
        f.write(f"Base directory: {base_dir}\n\n")

        # K-effective results
        f.write("K-EFFECTIVE STATISTICS\n")
        f.write("=" * 30 + "\n\n")

        if 'keff_values' in stats:
            # Actual k-effective values
            keff_vals = stats['keff_values']
            f.write("K-effective Values:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Minimum:        {keff_vals['min']:.6f}\n")
            f.write(f"  Maximum:        {keff_vals['max']:.6f}\n")
            f.write(f"  Range:          {keff_vals['max'] - keff_vals['min']:.6f}\n")
            f.write(f"  Mean:           {keff_vals['mean']:.6f}\n")
            f.write(f"  Std Deviation:  {keff_vals['std']:.6f}\n")
            f.write(f"  Sample Count:   {keff_vals['count']}\n\n")

            # K-effective standard deviations
            keff_stds = stats['keff_stds']
            f.write("K-effective Standard Deviations:\n")
            f.write("-" * 35 + "\n")
            f.write(f"  Minimum:        {keff_stds['min']:.6e}\n")
            f.write(f"  Maximum:        {keff_stds['max']:.6e}\n")
            f.write(f"  Range:          {keff_stds['max'] - keff_stds['min']:.6e}\n")
            f.write(f"  Mean:           {keff_stds['mean']:.6e}\n")
            f.write(f"  Std Deviation:  {keff_stds['std']:.6e}\n")
            f.write(f"  Sample Count:   {keff_stds['count']}\n\n")

            # K-effective uncertainty percentages
            keff_pcts = stats['keff_uncertainty_pct']
            f.write("K-effective Uncertainty Percentages (std_dev/keff * 100):\n")
            f.write("-" * 55 + "\n")
            f.write(f"  Minimum:        {keff_pcts['min']:.6f}%\n")
            f.write(f"  Maximum:        {keff_pcts['max']:.6f}%\n")
            f.write(f"  Range:          {keff_pcts['max'] - keff_pcts['min']:.6f}%\n")
            f.write(f"  Mean:           {keff_pcts['mean']:.6f}%\n")
            f.write(f"  Std Deviation:  {keff_pcts['std']:.6f}%\n")
            f.write(f"  Sample Count:   {keff_pcts['count']}\n\n")
        else:
            f.write("No k-effective data found\n\n")

        # Flux results (all positions combined)
        f.write("FLUX STATISTICS (ALL POSITIONS COMBINED)\n")
        f.write("=" * 45 + "\n\n")

        flux_types = ['total_flux', 'thermal', 'epithermal', 'fast']

        for flux_type in flux_types:
            flux_name = flux_type.replace('_', ' ').title()
            f.write(f"{flux_name.upper()} FLUX:\n")
            f.write("=" * (len(flux_name) + 6) + "\n\n")

            # Actual flux values
            value_key = f'{flux_type}_values'
            if value_key in stats:
                flux_vals = stats[value_key]
                f.write(f"{flux_name} Values:\n")
                f.write("-" * (len(flux_name) + 8) + "\n")
                f.write(f"  Minimum:        {flux_vals['min']:.6e}\n")
                f.write(f"  Maximum:        {flux_vals['max']:.6e}\n")
                f.write(f"  Range:          {flux_vals['max'] - flux_vals['min']:.6e}\n")
                f.write(f"  Mean:           {flux_vals['mean']:.6e}\n")
                f.write(f"  Std Deviation:  {flux_vals['std']:.6e}\n")
                f.write(f"  Sample Count:   {flux_vals['count']}\n\n")

                # Flux standard deviations
                std_key = f'{flux_type}_stds'
                if std_key in stats:
                    flux_stds = stats[std_key]
                    f.write(f"{flux_name} Standard Deviations:\n")
                    f.write("-" * (len(flux_name) + 23) + "\n")
                    f.write(f"  Minimum:        {flux_stds['min']:.6e}\n")
                    f.write(f"  Maximum:        {flux_stds['max']:.6e}\n")
                    f.write(f"  Range:          {flux_stds['max'] - flux_stds['min']:.6e}\n")
                    f.write(f"  Mean:           {flux_stds['mean']:.6e}\n")
                    f.write(f"  Std Deviation:  {flux_stds['std']:.6e}\n")
                    f.write(f"  Sample Count:   {flux_stds['count']}\n\n")

                # Flux uncertainty percentages
                pct_key = f'{flux_type}_uncertainty_pct'
                if pct_key in stats:
                    flux_pcts = stats[pct_key]
                    f.write(f"{flux_name} Uncertainty Percentages (std_dev/value * 100):\n")
                    f.write("-" * (len(flux_name) + 43) + "\n")
                    f.write(f"  Minimum:        {flux_pcts['min']:.6f}%\n")
                    f.write(f"  Maximum:        {flux_pcts['max']:.6f}%\n")
                    f.write(f"  Range:          {flux_pcts['max'] - flux_pcts['min']:.6f}%\n")
                    f.write(f"  Mean:           {flux_pcts['mean']:.6f}%\n")
                    f.write(f"  Std Deviation:  {flux_pcts['std']:.6f}%\n")
                    f.write(f"  Sample Count:   {flux_pcts['count']}\n\n")
            else:
                f.write(f"No {flux_name.lower()} flux data found\n\n")

    print(f"\nAnalysis complete! Results written to: {output_file}")
    return output_file

def find_parametric_folders():
    """Find all parametric simulation folders in the Integrated Reactor Model directory"""
    base_path = "Integrated Reactor Model"
    parametric_folders = []

    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and "parametric_simulation" in item:
                parametric_folders.append(item_path)

    return sorted(parametric_folders)

def select_parametric_folder():
    """Interactive selection of parametric simulation folder"""
    parametric_folders = find_parametric_folders()

    if not parametric_folders:
        print("No parametric simulation folders found in 'Integrated Reactor Model' directory!")
        return None

    print("\nAvailable parametric simulation folders:")
    print("=" * 40)

    for i, folder in enumerate(parametric_folders, 1):
        folder_name = os.path.basename(folder)
        print(f"{i}. {folder_name}")

    print("=" * 40)

    while True:
        try:
            choice = input(f"\nSelect folder to analyze (1-{len(parametric_folders)}): ").strip()
            choice_num = int(choice)

            if 1 <= choice_num <= len(parametric_folders):
                selected_folder = parametric_folders[choice_num - 1]
                print(f"\nSelected: {os.path.basename(selected_folder)}")
                return selected_folder
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(parametric_folders)}")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

if __name__ == "__main__":
    # Interactive folder selection
    selected_folder = select_parametric_folder()

    if selected_folder:
        print(f"\nStarting analysis of: {selected_folder}")
        analyze_parametric_results(selected_folder)
    else:
        print("No folder selected. Exiting.")
