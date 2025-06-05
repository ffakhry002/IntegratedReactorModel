"""
Main visualization script for all sampling methods.
Creates organized outputs in lattice/ and geometric/ subfolders.
UPDATED: Support for 6x6 restricted configurations

This is the refactored version that uses modular components from the
visualization_code/ folder. To set up:

1. Create a folder called 'visualization_code' in the same directory
2. Place all the module files in that folder:
   - __init__.py
   - data_loader.py
   - config_visualizer.py
   - parameter_plots.py
   - analysis_plots.py
3. Run this script to generate all visualizations

The original monolithic script has been split into focused modules for
better maintainability and code organization.
"""

import os
import sys
from pathlib import Path
import pickle

# Get script directory for relative paths (allows running from outside directory)
SCRIPT_DIR = Path(__file__).parent.absolute()

# Add the script directory to Python path to ensure imports work
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualization_code.data_loader import (
    load_core_configurations, load_physics_parameters, load_all_results,
    get_method_colors, get_method_lists, load_physics_parameters_for_method,
    load_configurations_for_method, detect_6x6_mode
)
from visualization_code.config_visualizer import create_method_visualization
from visualization_code.parameter_plots import (
    plot_physics_parameters_comparison,
    create_diversity_comparison_by_type
)
from visualization_code.analysis_plots import (
    create_method_visualizations,
    create_combined_analysis,
    create_summary_statistics
)


def main():
    """Main visualization function with organized output structure."""
    # Detect if we're in 6x6 mode
    is_6x6 = detect_6x6_mode()
    mode_str = " (6x6 Central Square)" if is_6x6 else ""

    print(f"COMPREHENSIVE SAMPLING VISUALIZATION{mode_str}")
    print("="*80)

    if is_6x6:
        print("✓ Detected 6x6 restriction mode")
        print("  - Visualizations will show configurations restricted to central 6x6 square")
        print("  - Physics parameters calculated considering full 8x8 grid context")
        print()

    # Get method lists and colors
    lattice_methods, geometric_methods = get_method_lists()
    all_methods = lattice_methods + geometric_methods
    method_colors = get_method_colors()

    # Ensure visualization directories exist
    (SCRIPT_DIR / 'visualizations/lattice').mkdir(parents=True, exist_ok=True)
    (SCRIPT_DIR / 'visualizations/geometric').mkdir(parents=True, exist_ok=True)

    # Load all available results
    results_data, samples_data = load_all_results(all_methods)

    if not samples_data:
        print("No sampling results found! Run sampling first.")
        return

    print(f"\nFound results for {len(samples_data)} methods")

    # Load core configurations
    try:
        configurations, irradiation_sets = load_core_configurations(use_6x6=is_6x6)
        print(f"✓ Loaded configuration data ({len(configurations):,} configs)")
    except FileNotFoundError as e:
        print(f"✗ Missing required data: {e}")
        return

    # Create comprehensive visualizations
    print("\nCreating visualizations...")

    # Filter methods to only those that have results
    available_lattice_methods = [m for m in lattice_methods if m in samples_data]
    available_geometric_methods = [m for m in geometric_methods if m in samples_data]

    # 1. Method comparison plots (separate for lattice and geometric)
    if available_lattice_methods or available_geometric_methods:
        create_diversity_comparison_by_type(
            samples_data, available_lattice_methods,
            available_geometric_methods, method_colors
        )

    # 2. Create parameter comparison plots for each type
    # Load appropriate physics parameters for lattice methods
    if available_lattice_methods:
        # Check max index for lattice methods to determine which physics params to load
        max_idx_lattice = 0
        for method in available_lattice_methods:
            if method in samples_data:
                indices = samples_data[method]['selected_indices']
                if indices:
                    max_idx_lattice = max(max_idx_lattice, max(indices))

        physics_params_lattice, param_type = load_physics_parameters_for_method(
            available_lattice_methods[0] if available_lattice_methods else None,
            max_idx_lattice,
            use_6x6=is_6x6
        )
        print(f"✓ Loaded {param_type} physics parameters for lattice methods (max index: {max_idx_lattice})")

        plot_physics_parameters_comparison(
            available_lattice_methods, physics_params_lattice,
            samples_data, str(SCRIPT_DIR / 'visualizations/lattice')
        )

    # Load appropriate physics parameters for geometric methods
    if available_geometric_methods:
        # Check max index for geometric methods
        max_idx = 0
        for method in available_geometric_methods:
            if method in samples_data:
                indices = samples_data[method]['selected_indices']
                if indices:
                    max_idx = max(max_idx, max(indices))

        physics_params_geometric, param_type = load_physics_parameters_for_method(
            available_geometric_methods[0] if available_geometric_methods else None,
            max_idx,
            use_6x6=is_6x6
        )
        print(f"✓ Loaded {param_type} physics parameters for geometric methods ({len(physics_params_geometric):,} configs)")

        plot_physics_parameters_comparison(
            available_geometric_methods, physics_params_geometric,
            samples_data, str(SCRIPT_DIR / 'visualizations/geometric')
        )

    # 3. Individual method visualizations organized by type
    for method in available_lattice_methods:
        if method in method_colors:
            # Create dedicated subfolder for each lattice method
            method_dir = SCRIPT_DIR / f'visualizations/lattice/{method}'
            method_dir.mkdir(parents=True, exist_ok=True)
            output_dir = str(method_dir)

            # Load appropriate configs and physics params based on this method's indices
            max_idx = max(samples_data[method]['selected_indices']) if samples_data[method]['selected_indices'] else 0
            configurations, irradiation_sets, config_type = load_configurations_for_method(method, max_idx, use_6x6=is_6x6)
            physics_params, param_type = load_physics_parameters_for_method(method, max_idx, use_6x6=is_6x6)

            print(f"  Using {config_type} configurations and {param_type} physics params for {method}")

            create_method_visualizations(
                method, samples_data[method], configurations,
                physics_params, irradiation_sets, method_colors[method],
                output_dir=output_dir
            )

    for method in available_geometric_methods:
        if method in method_colors:
            # Create dedicated subfolder for each geometric method
            method_dir = SCRIPT_DIR / f'visualizations/geometric/{method}'
            method_dir.mkdir(parents=True, exist_ok=True)
            output_dir = str(method_dir)

            # Load appropriate configs and physics params for this geometric method
            max_idx = max(samples_data[method]['selected_indices']) if samples_data[method]['selected_indices'] else 0
            configurations, irradiation_sets, config_type = load_configurations_for_method(method, max_idx, use_6x6=is_6x6)
            physics_params, _ = load_physics_parameters_for_method(method, max_idx, use_6x6=is_6x6)

            print(f"  Using {config_type} configurations for {method}")

            create_method_visualizations(
                method, samples_data[method], configurations,
                physics_params, irradiation_sets, method_colors[method],
                output_dir=output_dir
            )

    # 4. Combined analysis plots
    if samples_data:
        # For combined analysis, we need to handle mixed physics parameters and configurations carefully
        # Load both sets if needed
        physics_params_reduced = load_physics_parameters(use_6x6=is_6x6)
        configurations_reduced, irradiation_sets_reduced = load_core_configurations(use_6x6=is_6x6)

        physics_params_full = None
        configurations_full = None
        irradiation_sets_full = None

        # Check if any method needs full params/configs
        needs_full = False
        threshold = 34119 if not is_6x6 else 9999  # Adjust based on actual 6x6 counts

        for method in samples_data:
            if method in samples_data:
                max_idx = max(samples_data[method]['selected_indices']) if samples_data[method]['selected_indices'] else 0
                if max_idx >= threshold:
                    needs_full = True
                    break

        if needs_full:
            suffix = "_6x6" if is_6x6 else ""
            full_physics_path = SCRIPT_DIR / f'output/data/physics_parameters_full{suffix}.pkl'
            if full_physics_path.exists():
                try:
                    with open(full_physics_path, 'rb') as f:
                        data = pickle.load(f)
                        physics_params_full = data['parameters']
                except:
                    pass

            full_configs_path = SCRIPT_DIR / f'output/data/all_configurations_before_symmetry{suffix}.pkl'
            if full_configs_path.exists():
                try:
                    with open(full_configs_path, 'rb') as f:
                        data = pickle.load(f)
                        configurations_full = data['configurations']
                        irradiation_sets_full = data['irradiation_sets']
                except:
                    pass

        # Use full params if available and needed, otherwise reduced
        physics_params_combined = physics_params_full if physics_params_full else physics_params_reduced
        configurations_combined = configurations_full if configurations_full else configurations_reduced

        create_combined_analysis(
            samples_data, configurations_combined, physics_params_combined,
            available_lattice_methods, available_geometric_methods,
            method_colors
        )

    # 5. Create overall physics parameters comparison
    if samples_data:
        all_available_methods = available_lattice_methods + available_geometric_methods

        # For combined plot, check which params to use
        max_idx = 0
        for method in all_available_methods:
            if method in samples_data:
                indices = samples_data[method]['selected_indices']
                if indices:
                    max_idx = max(max_idx, max(indices))

        physics_params_all, param_type = load_physics_parameters_for_method(None, max_idx, use_6x6=is_6x6)
        print(f"✓ Using {param_type} physics parameters for combined plot")

        plot_physics_parameters_comparison(
            all_available_methods, physics_params_all,
            samples_data, str(SCRIPT_DIR / 'visualizations')
        )

    # 6. Summary statistics
    create_summary_statistics(
        samples_data, available_lattice_methods,
        available_geometric_methods
    )

    print("\n" + "="*80)
    print(f"VISUALIZATION COMPLETE!{mode_str}")
    print("="*80)
    print("Generated visualizations in:")
    if available_lattice_methods:
        print("  visualizations/lattice/    - Lattice-based method plots")
    if available_geometric_methods:
        print("  visualizations/geometric/  - Geometric/physics method plots")
    if samples_data:
        print("  visualizations/            - Combined analysis plots")
    print("  + Comprehensive summary graphs for each method type")

    if is_6x6:
        print("\nNote: All visualizations reflect 6x6 central square restriction")


if __name__ == "__main__":
    main()
