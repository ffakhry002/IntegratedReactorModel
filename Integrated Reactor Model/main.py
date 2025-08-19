"""
Main script to run the integrated reactor simulation.
"""

import os
import sys
import shutil
import copy
from eigenvalue.run import run_eigenvalue
from Reactor.geometry import plot_geometry
from ThermalHydraulics.TH_refactored import THSystem
from plotting.plotall import plot_all
from depletion.run_depletion import run_all_depletions
from inputs import inputs
from eigenvalue.parametric_study import run_parametric_study

def cleanup_all_pycache():
    """Remove all __pycache__ directories in the entire project structure.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk through all directories in the project
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                # print(f"Removed: {pycache_path}")
            except Exception as e:
                print(f"Error removing {pycache_path}: {e}")

def run_additional_th_calculations(root_dir, dirs, th_subdirs, inputs_dict=None):
    """Run additional thermal hydraulics calculations with different power profiles.

    Parameters
    ----------
    root_dir : str
        Root directory of the project
    dirs : dict
        Dictionary of subdirectories for output
    th_subdirs : dict
        Dictionary of thermal hydraulics subdirectories
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    None
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    # Get element type from inputs
    is_element_level = inputs_dict.get('element_level_power_tallies', False)
    if is_element_level:
        if inputs_dict['assembly_type'] == 'Pin':
            element_type = "pin"
        else:
            element_type = "plate"
    else:
        element_type = "assembly"

    # Path to the power distribution CSV
    power_csv = os.path.join(dirs['power_plots'], f'detailed_{element_type}_power_distribution.csv')

    if not os.path.exists(power_csv):
        print(f"\nPower distribution CSV not found at {power_csv}. Skipping additional TH calculations.")
        return

    # Run TH calculations for hot element
    print("\nRunning TH with hot element power profile...")
    # Create a deep copy of inputs and modify for hot element
    hot_inputs = copy.deepcopy(inputs_dict)
    # Create a new THSystem with hot element power profile
    th_system_hot = THSystem(hot_inputs)
    th_system_hot.reactor_power.power_source = 'HOT_ELEMENT'
    th_system_hot.reactor_power.csv_path = power_csv
    thermal_state_hot = th_system_hot.calculate_temperature_distribution()
    th_system_hot.write_results(th_subdirs['hot_element'])

    # Run TH calculations for core average
    print("\nRunning TH with core average power profile...")
    # Create a deep copy of inputs and modify for core average
    avg_inputs = copy.deepcopy(inputs_dict)
    # Create a new THSystem with core average power profile
    th_system_avg = THSystem(avg_inputs)
    th_system_avg.reactor_power.power_source = 'CORE_AVERAGE'
    th_system_avg.reactor_power.csv_path = power_csv
    thermal_state_avg = th_system_avg.calculate_temperature_distribution()
    th_system_avg.write_results(th_subdirs['core_average'])

    print("\nCompleted all thermal hydraulics analyses.")

def main():
    """Run the complete reactor simulation workflow.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Check if parametric study mode is enabled
    if inputs.get('parametric_study', False):
        print("Parametric study mode enabled. Running parametric study...")
        run_parametric_study()
        return

    # Clean up all __pycache__ directories first
    print("\nCleaning up __pycache__ directories...")
    cleanup_all_pycache()

    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Create simulation directory structure
    print("\nCreating simulation directory structure...")
    # Create main simulation directory
    sim_dir = os.path.join(root_dir, 'simulation_data')
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir)

    # Create subdirectories
    dirs = {
        'geometry_materials': os.path.join(sim_dir, 'Geometry_and_Materials'),
        'thermal_hydraulics': os.path.join(sim_dir, 'ThermalHydraulics'),
        'transport_data': os.path.join(sim_dir, 'transport_data'),
        'flux_plots': os.path.join(sim_dir, 'flux_plots'),
        'power_plots': os.path.join(sim_dir, 'power_plots'),
        'depletion_data': os.path.join(sim_dir, 'depletion_data'),
        'depletion_plots': os.path.join(sim_dir, 'depletion_plots')
    }

    # Create each subdirectory
    for dir_path in dirs.values():
        os.makedirs(dir_path)

    # Create TH scenario directories
    th_scenarios = ['cosine_calculation', 'hot_element', 'core_average']
    th_subdirs = {}
    for scenario in th_scenarios:
        scenario_dir = os.path.join(dirs['thermal_hydraulics'], scenario)
        os.makedirs(scenario_dir)
        th_subdirs[scenario] = scenario_dir

    # Run geometry and materials generation
    print("\nGenerating geometry and materials...")
    plot_geometry(dirs['geometry_materials'], inputs_dict=inputs)

    # Run thermal hydraulics with cosine approximation
    print("\nRunning thermal hydraulics analysis with cosine approximation...")
    th_system = THSystem(inputs)
    thermal_state = th_system.calculate_temperature_distribution()
    th_system.write_results(th_subdirs['cosine_calculation'])

    # Run OpenMC simulation
    print("\nRunning OpenMC simulation...")
    k_eff, k_std = run_eigenvalue(inputs_dict=inputs)
    print(f"\nSimulation completed successfully!")
    print(f"k-effective = {k_eff:.6f} ± {k_std:.6f}")

    # Run depletion calculations if enabled
    any_depletion_enabled = any(v for k, v in inputs.items() if k.startswith('deplete_'))
    if any_depletion_enabled:
        print("\nRunning depletion calculations...")
        depletion_results = run_all_depletions(output_dir=dirs['depletion_data'], inputs_dict=inputs)
    else:
        print("\nNo depletion calculations enabled in inputs")

    # Check if power tallies are enabled
    if inputs.get('tally_power', True):
        # Generate all plots (including power plots)
        print("\nGenerating plots...")
        plot_all(plot_dir=dirs['flux_plots'], depletion_plot_dir=dirs['depletion_plots'], power_plot_dir=dirs['power_plots'], inputs_dict=inputs)

        # Run additional thermal hydraulics calculations with different power profiles
        run_additional_th_calculations(root_dir, dirs, th_subdirs, inputs_dict=inputs)
    else:
        # Generate only flux and depletion plots (no power plots)
        print("\nGenerating flux and depletion plots only...")
        plot_all(plot_dir=dirs['flux_plots'], depletion_plot_dir=dirs['depletion_plots'], power_plot_dir=None, inputs_dict=inputs)
        print("\nSkipping additional thermal hydraulics calculations (power tallies disabled)")

    # Final cleanup of any new __pycache__ directories created during the run
    print("\nFinal cleanup of __pycache__ directories...")
    cleanup_all_pycache()

if __name__ == "__main__":
    main()
