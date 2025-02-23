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

def plot_all(flux_plot_dir=None, depletion_plot_dir=None):
    """Plot all distributions from the simulation results.

    Parameters
    ----------
    flux_plot_dir : str, optional
        Directory to save flux-related plots to. If not provided, will use default locations
        based on execution context.
    depletion_plot_dir : str, optional
        Directory to save depletion-related plots to. If not provided, will use default locations
        based on execution context.
    """
    # Get root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if running directly by looking at the script name
    running_directly = os.path.basename(sys.argv[0]) == 'plotall.py'

    # Determine paths based on context
    if not running_directly and os.path.exists(os.path.join(root_dir, 'simulation_data')):
        # Running from main.py
        statepoint_path = os.path.join(root_dir, 'simulation_data', 'transport_data', 'statepoint.eigenvalue.h5')
        flux_plot_dir = flux_plot_dir or os.path.join(root_dir, 'simulation_data', 'flux_plots')
        depletion_plot_dir = depletion_plot_dir or os.path.join(root_dir, 'simulation_data', 'depletion_plots')
        depletion_dir = os.path.join(root_dir, 'simulation_data', 'depletion_data')
    else:
        # Running directly
        statepoint_path = os.path.join(root_dir, 'execution', 'Output', 'statepoint.eigenvalue.h5')
        flux_plot_dir = flux_plot_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flux_plots')
        depletion_plot_dir = depletion_plot_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depletion_plots')
        depletion_dir = os.path.join(root_dir, 'depletion', 'outputs')

    if not os.path.exists(statepoint_path):
        print(f"Transport statepoint file not found at {statepoint_path}")
        return
    else:
        print(f"Loading transport statepoint file: {statepoint_path}")
        sp = openmc.StatePoint(statepoint_path)

    # Create flux plot directory if it doesn't exist
    if os.path.exists(flux_plot_dir):
        print("\nCleaning up old flux plotting files...")
        import shutil
        shutil.rmtree(flux_plot_dir)
    os.makedirs(flux_plot_dir, exist_ok=True)

    # Get power from inputs, default to 1 MW if not specified
    power_mw = inputs.get('core_power', 1.0)
    print(f"Using reactor power of {power_mw} MW for normalization")

    # Generate all flux-related plots
    try:
        print("\nGenerating entropy convergence plot...")
        plot_entropy(sp, flux_plot_dir)
    except Exception as e:
        print(f"Error generating entropy plot: {str(e)}")

    # Check if there are any irradiation positions in the core layout
    has_irradiation = any('I' in cell for row in inputs['core_lattice'] for cell in row)
    if has_irradiation:
        try:
            print("\nGenerating flux trap plots...")
            plot_flux_trap_distributions(sp, power_mw, flux_plot_dir)
        except Exception as e:
            print(f"Error generating flux trap plots: {str(e)}")
    else:
        print("\nSkipping flux trap plots (no irradiation positions in core)")

    try:
        print("\nGenerating flux maps...")
        plot_flux_maps(sp, flux_plot_dir)
    except Exception as e:
        print(f"Error generating flux maps: {str(e)}")

    try:
        print("\nGenerating normalized flux profiles...")
        plot_normalized_flux_profiles(sp, flux_plot_dir)
    except Exception as e:
        print(f"Error generating normalized flux profiles: {str(e)}")

    print(f"\nFlux plots have been saved to: {flux_plot_dir}")

    # Generate depletion plots if depletion results exist
    if os.path.exists(depletion_dir):
        # Create depletion plot directory if it doesn't exist
        if os.path.exists(depletion_plot_dir):
            print("\nCleaning up old depletion plotting files...")
            import shutil
            shutil.rmtree(depletion_plot_dir)
        os.makedirs(depletion_plot_dir, exist_ok=True)

        try:
            print("\nGenerating depletion results plots...")
            plot_depletion_results(depletion_plot_dir, root_dir=root_dir, depletion_dir=depletion_dir)
            print(f"\nDepletion plots have been saved to: {depletion_plot_dir}")
        except Exception as e:
            print(f"Error generating depletion results plots: {str(e)}")
    else:
        print("\nSkipping depletion plots (no depletion results found)")

def main():
    """Run all plotting functions."""
    try:
        plot_all()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
