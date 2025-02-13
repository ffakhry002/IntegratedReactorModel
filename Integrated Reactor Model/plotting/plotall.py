"""
Main script to run all plotting functions.
"""

import os
import sys
import openmc

# Add root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from inputs import inputs
from plotting.functions.flux_traps import plot_flux_trap_distributions
from plotting.functions.flux_maps import plot_flux_maps
from plotting.functions.normalized_flux_profiles import plot_normalized_flux_profiles
from plotting.functions.entropy import plot_entropy
from plotting.functions.depletion import plot_depletion_results

def plot_all_flux_distributions():
    """Plot all flux distributions from the statepoint file.

    The function automatically detects if it's being run from main.py or directly:
    - If run from main.py: uses statepoint from simulation_data/xml_and_h5/
    - If run directly: uses statepoint from execution/Output/

    Plots are saved in:
    - If run from main.py: simulation_data/flux_plots/
    - If run directly: plotting/plots/
    """
    # Get root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if running directly by looking at the script name
    running_directly = os.path.basename(sys.argv[0]) == 'plotall.py'

    if not running_directly and os.path.exists(os.path.join(root_dir, 'simulation_data')):
        statepoint_path = os.path.join(root_dir, 'simulation_data', 'xml_and_h5', 'statepoint.eigenvalue.h5')
        plot_dir = os.path.join(root_dir, 'simulation_data', 'flux_plots')
    else:
        statepoint_path = os.path.join(root_dir, 'execution', 'Output', 'statepoint.eigenvalue.h5')
        # When running directly, save to plotting/plots directory
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

    if not os.path.exists(statepoint_path):
        raise FileNotFoundError(f"Statepoint file not found at {statepoint_path}")

    # Create plot directory if it doesn't exist

    if os.path.exists(plot_dir):
        print("\nCleaning up old plotting files...")
        import shutil
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    # Load statepoint file
    print(f"Loading statepoint file: {statepoint_path}")
    sp = openmc.StatePoint(statepoint_path)

    # Get power from inputs, default to 1 MW if not specified
    power_mw = inputs.get('core_power', 1.0)
    print(f"Using reactor power of {power_mw} MW for normalization")

    # Generate all plots
    try:
        print("\nGenerating entropy convergence plot...")
        plot_entropy(sp, plot_dir)
    except Exception as e:
        print(f"Error generating entropy plot: {str(e)}")

    # Check if there are any irradiation positions in the core layout
    has_irradiation = any('I' in cell for row in inputs['core_lattice'] for cell in row)
    if has_irradiation:
        try:
            print("\nGenerating flux trap plots...")
            plot_flux_trap_distributions(sp, power_mw, plot_dir)
        except Exception as e:
            print(f"Error generating flux trap plots: {str(e)}")
    else:
        print("\nSkipping flux trap plots (no irradiation positions in core)")

    try:
        print("\nGenerating flux maps...")
        plot_flux_maps(sp, plot_dir)
    except Exception as e:
        print(f"Error generating flux maps: {str(e)}")

    try:
        print("\nGenerating normalized flux profiles...")
        plot_normalized_flux_profiles(sp, plot_dir)
    except Exception as e:
        print(f"Error generating normalized flux profiles: {str(e)}")

    try:
        print("\nGenerating depletion results plots...")
        plot_depletion_results(plot_dir)
    except Exception as e:
        print(f"Error generating depletion results plots: {str(e)}")

    print(f"\nPlots have been saved to: {plot_dir}")

def main():
    """Run all plotting functions."""
    try:
        plot_all_flux_distributions()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
