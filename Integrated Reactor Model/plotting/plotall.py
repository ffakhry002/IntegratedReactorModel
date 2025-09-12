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
from plotting.functions.axial_flux_energy import plot_axial_flux_energy_breakdown
from plotting.functions.entropy import plot_entropy
from plotting.functions.depletion import plot_depletion_results, plot_nuclide_evolution
from plotting.functions.power import plot_power_distributions, plot_2d_power_map

def plot_all(plot_dir=None, depletion_plot_dir=None, power_plot_dir=None, inputs_dict=None):
    """Plot all distributions from the simulation results.

    Parameters
    ----------
    plot_dir : str, optional
        Directory to save flux plots to. If not provided, will use default locations
        based on execution context.
    depletion_plot_dir : str, optional
        Directory to save depletion-related plots to. If not provided, will use default locations
        based on execution context.
    power_plot_dir : str, optional
        Directory to save power plots to. If not provided, will use default locations
        based on execution context.
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Get root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if running directly by looking at the script name
    running_directly = os.path.basename(sys.argv[0]) == 'plotall.py'

    # Check if we're in a parametric study run directory
    current_dir = os.getcwd()
    if 'parametric_simulation_' in current_dir and 'run_' in current_dir:
        # We're in a parametric study run directory - use the passed directories
        statepoint_path = os.path.join(current_dir, 'transport_data', 'statepoint.eigenvalue.h5')
        flux_plot_dir = plot_dir  # Should always be provided in parametric mode
        power_plot_dir = power_plot_dir or os.path.join(current_dir, 'power_plots')  # Use parametric structure
        depletion_plot_dir = depletion_plot_dir  # Should be provided in parametric mode
        depletion_dir = os.path.join(current_dir, 'depletion_data')  # Use parametric structure
    elif not running_directly and os.path.exists(os.path.join(root_dir, 'simulation_data')):
        # Running from main.py
        base_dir = os.path.join(root_dir, 'simulation_data')
        statepoint_path = os.path.join(base_dir, 'transport_data', 'statepoint.eigenvalue.h5')
        flux_plot_dir = plot_dir or os.path.join(base_dir, 'flux_plots')
        power_plot_dir = power_plot_dir or os.path.join(base_dir, 'power_plots')
        depletion_plot_dir = depletion_plot_dir or os.path.join(base_dir, 'depletion_plots')
        depletion_dir = os.path.join(base_dir, 'depletion_data')
    else:
        # Running directly - put everything under eigenvalue folder
        base_dir = os.path.join(root_dir, 'eigenvalue')
        statepoint_path = os.path.join(base_dir, 'Output', 'statepoint.eigenvalue.h5')
        plots_dir = os.path.join(base_dir, 'plots')
        flux_plot_dir = plot_dir or os.path.join(plots_dir, 'flux_plots')
        power_plot_dir = power_plot_dir or os.path.join(plots_dir, 'power_plots')
        depletion_plot_dir = depletion_plot_dir or os.path.join(plots_dir, 'depletion_plots')
        depletion_dir = os.path.join(root_dir, 'depletion', 'outputs')

    if not os.path.exists(statepoint_path):
        print(f"Transport statepoint file not found at {statepoint_path}")
        return
    else:
        print(f"Loading transport statepoint file: {statepoint_path}")
        sp = openmc.StatePoint(statepoint_path)

    # Clean up and create plot directories
    for directory in [flux_plot_dir, power_plot_dir]:
        if directory is not None and os.path.exists(directory):
            print(f"\nCleaning up old plotting files in {directory}...")
            import shutil
            shutil.rmtree(directory)
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

    # Get power from inputs, default to 1 MW if not specified
    power_mw = inputs_dict.get('core_power', 1.0)
    print(f"Using reactor power of {power_mw} MW for normalization")

    # Generate all flux-related plots
    try:
        print("\nGenerating entropy convergence plot...")
        plot_entropy(sp, flux_plot_dir, inputs_dict)
    except Exception as e:
        print(f"Error generating entropy plot: {str(e)}")

    # Check if there are any irradiation positions in the core layout
    has_irradiation = any('I' in cell for row in inputs_dict['core_lattice'] for cell in row)
    if has_irradiation:
        try:
            print("\nGenerating flux trap plots...")
            plot_flux_trap_distributions(sp, power_mw, flux_plot_dir, inputs_dict)
        except Exception as e:
            print(f"Error generating flux trap plots: {str(e)}")
    else:
        print("\nSkipping flux trap plots (no irradiation positions in core)")

    try:
        print("\nGenerating flux maps...")
        plot_flux_maps(sp, flux_plot_dir, inputs_dict)
    except Exception as e:
        print(f"Error generating flux maps: {str(e)}")

    try:
        print("\nGenerating normalized flux profiles...")
        plot_normalized_flux_profiles(sp, flux_plot_dir, inputs_dict)
    except Exception as e:
        print(f"Error generating normalized flux profiles: {str(e)}")

    # # Check if there are any irradiation positions for axial energy breakdown
    # if has_irradiation:
    #     try:
    #         print("\nGenerating axial flux energy breakdown plots...")
    #         plot_axial_flux_energy_breakdown(sp, flux_plot_dir, inputs_dict)
    #     except Exception as e:
    #         print(f"Error generating axial flux energy breakdown plots: {str(e)}")
    # else:
    #     print("\nSkipping axial flux energy breakdown plots (no irradiation positions in core)")

    # Only generate power plots if power tallies are enabled
    if power_plot_dir is not None:
        try:
            print("\nGenerating power distribution plots...")
            plot_power_distributions(sp, power_plot_dir, inputs_dict)
        except Exception as e:
            print(f"Error generating power distribution plots: {str(e)}")

        try:
            print("\nGenerating 2D power map...")
            plot_2d_power_map(sp, power_plot_dir, inputs_dict)
        except Exception as e:
            print(f"Error generating 3D power map: {str(e)}")

        print(f"\nPower plots have been saved to: {power_plot_dir}")
    else:
        print("\nSkipping power plots (power tallies disabled)")

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
            print("\nGenerating depletion plots...")
            plot_depletion_results(depletion_plot_dir, root_dir, depletion_dir, inputs_dict)
            plot_nuclide_evolution(depletion_plot_dir, root_dir, depletion_dir, inputs_dict)
            print(f"\nDepletion plots have been saved to: {depletion_plot_dir}")
        except Exception as e:
            print(f"Error generating depletion plots: {str(e)}")
    else:
        print("\nSkipping depletion plots (no depletion results found)")

def main():
    """Run all plotting functions.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Exits with status 1 if an error occurs during plotting.
    """
    try:
        plot_all()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
