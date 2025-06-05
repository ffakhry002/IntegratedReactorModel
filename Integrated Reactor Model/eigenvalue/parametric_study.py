"""
Parametric study functionality for the integrated reactor model.
"""

import os
import shutil
import copy
import time
import sys
from datetime import datetime

# Add the parent directory to the path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from inputs import inputs, calculate_derived_values
from run_dictionaries import all_runs
from eigenvalue.run import run_eigenvalue
from Reactor.geometry import plot_geometry
from ThermalHydraulics.TH_refactored import THSystem
from plotting.plotall import plot_all
from depletion.run_depletion import run_all_depletions

def create_parametric_directory():
    """Create the main parametric study directory with timestamp."""
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_dir_name = f"parametric_simulation_{current_date}"

    # Get the root directory (parent of eigenvalue where this script is located)
    script_dir = parent_dir  # Use the parent_dir we already calculated
    param_dir = os.path.join(script_dir, param_dir_name)

    if os.path.exists(param_dir):
        shutil.rmtree(param_dir)
    os.makedirs(param_dir)

    return param_dir

def write_base_inputs_log(param_dir, base_inputs):
    """Write the base inputs to a log file."""
    log_file = os.path.join(param_dir, "parametric_study_log.txt")

    # Get assembly type to determine which geometry parameters to show
    assembly_type = base_inputs.get('assembly_type', 'Pin')

    # Define units for parameters
    units = {
        'core_power': 'MW',
        'tank_radius': 'm',
        'reflector_thickness': 'm',
        'bioshield_thickness': 'm',
        'bottom_bioshield_thickness': 'm',
        'bottom_reflector_thickness': 'm',
        'feed_thickness': 'm',
        'plenum_height': 'm',
        'top_reflector_thickness': 'm',
        'top_bioshield_thickness': 'm',
        'fuel_height': 'm',
        # Pin geometry units
        'pin_pitch': 'm',
        'r_fuel': 'm',
        'r_clad_inner': 'm',
        'r_clad_outer': 'm',
        # Plate geometry units
        'fuel_meat_width': 'm',
        'fuel_plate_width': 'm',
        'fuel_plate_pitch': 'm',
        'fuel_meat_thickness': 'm',
        'clad_thickness': 'm',
        'clad_structure_width': 'm',
        'irradiation_clad_thickness': 'm',
        # Thermal hydraulics units
        'reactor_pressure': 'Pa',
        'flow_rate': 'm/s',
        'T_inlet': 'K',
        'input_power_density': 'kW/L',
        'max_linear_power': 'kW/m',
        'average_linear_power': 'kW/m',
        'thermal_cutoff': 'eV',
        'fast_cutoff': 'eV',
        # Enrichment units
        'n%': '%',
        'n%E': '%'
    }

    # Define pin-specific geometry parameters
    pin_geometry_params = [
        'pin_pitch', 'r_fuel', 'r_clad_inner', 'r_clad_outer',
        'n_side_pins', 'guide_tube_positions'
    ]

    # Define plate-specific geometry parameters
    plate_geometry_params = [
        'fuel_meat_width', 'fuel_plate_width', 'fuel_plate_pitch',
        'fuel_meat_thickness', 'clad_thickness', 'plates_per_assembly',
        'clad_structure_width'
    ]

    # Define the sections and their headers as they appear in inputs.py
    sections = {
        'Parametric Study Configuration': [
            'parametric_study'
        ],
        'Core Configuration': [
            'core_lattice', 'core_power', 'assembly_type'
        ],
        'Geometry Specifications': [
            'tank_radius', 'reflector_thickness', 'bioshield_thickness',
            'bottom_bioshield_thickness', 'bottom_reflector_thickness', 'feed_thickness',
            'plenum_height', 'top_reflector_thickness', 'top_bioshield_thickness', 'fuel_height'
        ],
        'Materials Configuration': [
            'coolant_type', 'clad_type', 'fuel_type', 'reflector_material', 'bioshield_material',
            'n%', 'n%E'
        ],
        'Thermal Hydraulics Parameters': [
            'reactor_pressure', 'flow_rate', 'T_inlet', 'input_power_density', 'max_linear_power',
            'average_linear_power', 'cos_curve_squeeze', 'CP_PD_MLP_ALP'
        ],
        'Irradiation Position Parameters': [
            'irradiation_clad', 'irradiation_clad_thickness', 'irradiation_cell_fill'
        ],
        'OpenMC Transport Parameters': [
            'batches', 'inactive', 'particles', 'energy_structure', 'thermal_cutoff', 'fast_cutoff',
            'power_tally_axial_segments', 'irradiation_axial_segments', 'core_mesh_dimension',
            'entropy_mesh_dimension', 'Core_Three_Group_Energy_Bins', 'tally_power', 'element_level_power_tallies'
        ],
        'Depletion Calculation Parameters': [
            'deplete_core', 'deplete_assembly', 'deplete_assembly_enhanced', 'deplete_element',
            'deplete_element_enhanced', 'depletion_timestep_units', 'depletion_timesteps',
            'depletion_particles', 'depletion_batches', 'depletion_inactive', 'depletion_integrator',
            'depletion_chain', 'depletion_nuclides'
        ],
        'Miscellaneous Settings': [
            'outputs_folder', 'pixels'
        ],
        'Derived Values': [
            'n_guide_tubes', 'num_assemblies'
        ]
    }

    # Add assembly-specific geometry section
    if assembly_type == 'Pin':
        sections['Pin Assembly Geometry'] = pin_geometry_params
    elif assembly_type == 'Plate':
        sections['Plate Assembly Geometry'] = plate_geometry_params

    def format_value_with_units(key, value):
        """Format value with units for better readability."""
        if key == 'core_lattice':
            # Format core_lattice as a multi-line array
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                lines = ["["]
                for i, row in enumerate(value):
                    comma = "," if i < len(value) - 1 else ""
                    lines.append(f"    {row}{comma}")
                lines.append("]")
                return "\n    ".join(lines)
        elif key == 'depletion_timesteps':
            # Format depletion_timesteps nicely
            if isinstance(value, list):
                lines = ["["]
                for i, step in enumerate(value):
                    comma = "," if i < len(value) - 1 else ""
                    lines.append(f"    {step}{comma}")
                lines.append("]")
                return "\n    ".join(lines)
        elif isinstance(value, list) and len(value) > 5:
            # Format long lists with line breaks
            return f"[\n    {', '.join(map(str, value))}\n]"
        elif isinstance(value, dict):
            # Format dictionaries nicely
            lines = ["{"]
            for k, v in value.items():
                lines.append(f"    '{k}': {v},")
            lines.append("}")
            return "\n".join(lines)

        # Add units if available
        formatted_value = str(value)
        if key in units:
            formatted_value += f" [{units[key]}]"

        return formatted_value

    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PARAMETRIC STUDY LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Study started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total number of runs: {len(all_runs)}\n")
        f.write("="*80 + "\n\n")

        f.write("BASE INPUTS:\n")
        f.write("="*80 + "\n\n")

        # Write inputs organized by sections
        written_keys = set()

        for section_name, section_keys in sections.items():
            f.write(f"{section_name}\n")
            f.write("="*79 + "\n")

            section_has_content = False
            for key in section_keys:
                if key in base_inputs:
                    formatted_value = format_value_with_units(key, base_inputs[key])
                    if key == 'core_lattice':
                        f.write(f"{key}:\n    {formatted_value}\n")
                    else:
                        f.write(f"{key}: {formatted_value}\n")
                    written_keys.add(key)
                    section_has_content = True

            if section_has_content:
                f.write("\n")

        f.write("="*80 + "\n\n")

    return log_file

def update_inputs_with_run_dict(base_inputs, run_dict):
    """Create a new inputs dictionary with modifications from run_dict."""
    # Start with a deep copy of base inputs
    modified_inputs = copy.deepcopy(base_inputs)

    # Apply changes from run_dict
    for key, value in run_dict.items():
        if key != "description":  # Don't update the description key
            modified_inputs[key] = value

    # Recalculate derived values if core_lattice or guide_tube_positions changed
    if "core_lattice" in run_dict or "guide_tube_positions" in run_dict:
        num_assemblies, n_guide_tubes = calculate_derived_values(
            modified_inputs["core_lattice"],
            modified_inputs["guide_tube_positions"]
        )
        modified_inputs["num_assemblies"] = num_assemblies
        modified_inputs["n_guide_tubes"] = n_guide_tubes

    return modified_inputs

def run_single_parametric_case(run_num, run_dict, param_dir, log_file):
    """Run a single parametric case."""
    print(f"\n{'='*60}")
    print(f"STARTING RUN {run_num}")
    print(f"{'='*60}")
    print(f"Description: {run_dict.get('description', 'No description')}")
    print(f"Modified parameters: {[k for k in run_dict.keys() if k != 'description']}")

    # Create run directory
    run_dir = os.path.join(param_dir, f"run_{run_num}")
    os.makedirs(run_dir)

    # Create subdirectories for this run
    subdirs = {
        'geometry_materials': os.path.join(run_dir, 'Geometry_and_Materials'),
        'thermal_hydraulics': os.path.join(run_dir, 'ThermalHydraulics'),
        'transport_data': os.path.join(run_dir, 'transport_data'),
        'flux_plots': os.path.join(run_dir, 'flux_plots'),
        'power_plots': os.path.join(run_dir, 'power_plots'),
        'depletion_data': os.path.join(run_dir, 'depletion_data'),
        'depletion_plots': os.path.join(run_dir, 'depletion_plots')
    }

    for dir_path in subdirs.values():
        os.makedirs(dir_path)

    # Create TH scenario directories
    th_scenarios = ['cosine_calculation', 'hot_element', 'core_average']
    th_subdirs = {}
    for scenario in th_scenarios:
        scenario_dir = os.path.join(subdirs['thermal_hydraulics'], scenario)
        os.makedirs(scenario_dir)
        th_subdirs[scenario] = scenario_dir

    # Update inputs for this run
    modified_inputs = update_inputs_with_run_dict(inputs, run_dict)

    try:
        # Save the current working directory
        original_cwd = os.getcwd()

        # Change to run directory for the simulation
        os.chdir(run_dir)

        # Run geometry and materials generation
        print("Generating geometry and materials...")
        plot_geometry(subdirs['geometry_materials'], inputs_dict=modified_inputs)

        # Run thermal hydraulics with cosine approximation
        print("Running thermal hydraulics analysis...")
        th_system = THSystem(modified_inputs)
        thermal_state = th_system.calculate_temperature_distribution()
        th_system.write_results(th_subdirs['cosine_calculation'])

        # Run OpenMC simulation
        print("Running OpenMC simulation...")
        k_eff, k_std = run_eigenvalue(inputs_dict=modified_inputs)

        print(f"Simulation completed successfully!")
        print(f"k-effective = {k_eff:.6f} ± {k_std:.6f}")

        # Run depletion calculations if enabled
        any_depletion_enabled = any(v for k, v in modified_inputs.items() if k.startswith('deplete_'))
        if any_depletion_enabled:
            print("Running depletion calculations...")
            depletion_results = run_all_depletions(output_dir=subdirs['depletion_data'], inputs_dict=modified_inputs)

        # Check if power tallies are enabled
        if modified_inputs.get('tally_power', True):
            # Generate all plots (including power plots)
            print("Generating plots...")
            plot_all(plot_dir=subdirs['flux_plots'], depletion_plot_dir=subdirs['depletion_plots'], power_plot_dir=subdirs['power_plots'], inputs_dict=modified_inputs)

            # Run additional thermal hydraulics calculations with different power profiles
            print("Running additional thermal hydraulics calculations...")
            run_additional_th_calculations(subdirs, th_subdirs, modified_inputs)
        else:
            # Generate only flux and depletion plots (no power plots)
            print("Power tallies disabled - generating flux and depletion plots only...")
            plot_all(plot_dir=subdirs['flux_plots'], depletion_plot_dir=subdirs['depletion_plots'], power_plot_dir=None, inputs_dict=modified_inputs)
            print("Skipping additional thermal hydraulics calculations (power tallies disabled)")

        # Log results
        log_run_results(log_file, run_num, run_dict, modified_inputs, k_eff, k_std, True)

        print(f"Run {run_num} completed successfully!")

    except Exception as e:
        print(f"Error in Run {run_num}: {str(e)}")
        log_run_results(log_file, run_num, run_dict, modified_inputs, None, None, False, str(e))

    finally:
        # Return to original directory
        os.chdir(original_cwd)

def run_additional_th_calculations(subdirs, th_subdirs, modified_inputs):
    """Run additional thermal hydraulics calculations with different power profiles."""
    # Get element type from inputs
    is_element_level = modified_inputs.get('element_level_power_tallies', False)
    if is_element_level:
        if modified_inputs['assembly_type'] == 'Pin':
            element_type = "pin"
        else:
            element_type = "plate"
    else:
        element_type = "assembly"

    # Path to the power distribution CSV
    power_csv = os.path.join(subdirs['power_plots'], f'detailed_{element_type}_power_distribution.csv')

    if not os.path.exists(power_csv):
        print(f"Power distribution CSV not found at {power_csv}. Skipping additional TH calculations.")
        return

    # Run TH calculations for hot element
    print("Running TH with hot element power profile...")
    hot_inputs = copy.deepcopy(modified_inputs)
    th_system_hot = THSystem(hot_inputs)
    th_system_hot.reactor_power.power_source = 'HOT_ELEMENT'
    th_system_hot.reactor_power.csv_path = power_csv
    thermal_state_hot = th_system_hot.calculate_temperature_distribution()
    th_system_hot.write_results(th_subdirs['hot_element'])

    # Run TH calculations for core average
    print("Running TH with core average power profile...")
    avg_inputs = copy.deepcopy(modified_inputs)
    th_system_avg = THSystem(avg_inputs)
    th_system_avg.reactor_power.power_source = 'CORE_AVERAGE'
    th_system_avg.reactor_power.csv_path = power_csv
    thermal_state_avg = th_system_avg.calculate_temperature_distribution()
    th_system_avg.write_results(th_subdirs['core_average'])

def log_run_results(log_file, run_num, run_dict, modified_inputs, k_eff, k_std, success, error_msg=None):
    """Log the results of a parametric run."""
    with open(log_file, 'a') as f:
        f.write(f"RUN {run_num}:\n")
        f.write("-"*40 + "\n")
        f.write(f"Description: {run_dict.get('description', 'No description')}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nModified Parameters:\n")
        for key, value in run_dict.items():
            if key != "description":
                f.write(f"  {key}: {value}\n")

        f.write(f"\nSuccess: {success}\n")
        if success and k_eff is not None:
            f.write(f"k-effective: {k_eff:.6f} ± {k_std:.6f}\n")

            # Extract irradiation flux data from results.txt
            results_file = os.path.join(os.getcwd(), 'results.txt')
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as rf:
                        lines = rf.readlines()

                    # Parse irradiation position results
                    in_irradiation_section = False
                    current_position = None

                    for line in lines:
                        line = line.strip()

                        if line.startswith("Irradiation Position Results:"):
                            in_irradiation_section = True
                            continue

                        if in_irradiation_section:
                            if line.startswith("Position I_"):
                                current_position = line.split(":")[0].replace("Position ", "")
                                continue

                            elif line.startswith("Total flux:"):
                                total_flux_str = line.split()[2]
                                continue

                            elif line.startswith("Thermal") and current_position:
                                parts = line.split()
                                thermal_flux = float(parts[1])
                                continue

                            elif line.startswith("Epithermal") and current_position:
                                parts = line.split()
                                epithermal_flux = float(parts[1])
                                continue

                            elif line.startswith("Fast") and current_position:
                                parts = line.split()
                                fast_flux = float(parts[1])

                                # Calculate total and percentages
                                total_flux = thermal_flux + epithermal_flux + fast_flux
                                thermal_percent = (thermal_flux / total_flux) * 100
                                epithermal_percent = (epithermal_flux / total_flux) * 100
                                fast_percent = (fast_flux / total_flux) * 100

                                # Write flux data
                                f.write(f"{current_position} Flux {total_flux:.2e} [{thermal_percent:.1f}% thermal, {epithermal_percent:.1f}% epithermal, {fast_percent:.1f}% fast]\n")

                                current_position = None
                                continue

                except Exception as e:
                    f.write(f"Error reading irradiation results: {e}\n")

        elif not success:
            f.write(f"Error: {error_msg}\n")

        f.write("\n" + "="*80 + "\n\n")

def run_parametric_study():
    """Main function to run the complete parametric study."""
    print("="*80)
    print("STARTING PARAMETRIC STUDY")
    print("="*80)
    print(f"Total number of runs: {len(all_runs)}")

    # Create main parametric directory
    param_dir = create_parametric_directory()
    print(f"Created parametric study directory: {param_dir}")

    # Write base inputs log
    log_file = write_base_inputs_log(param_dir, inputs)
    print(f"Base inputs logged to: {log_file}")

    # Run each parametric case
    for i, run_dict in enumerate(all_runs, 1):
        try:
            run_single_parametric_case(i, run_dict, param_dir, log_file)
        except Exception as e:
            print(f"Critical error in run {i}: {e}")
            # Log the critical error but continue with next run
            log_run_results(log_file, i, run_dict, {}, None, None, False, f"Critical error: {e}")

    # Final summary
    with open(log_file, 'a') as f:
        f.write("PARAMETRIC STUDY COMPLETED\n")
        f.write(f"Study finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

    print("="*80)
    print("PARAMETRIC STUDY COMPLETED")
    print("="*80)
    print(f"Results saved in: {param_dir}")
    print(f"Study log: {log_file}")
