from ..plottingcode.plotting_plate import plot_results_plate, calculate_cladding_temperature_profile
from ..plottingcode.plotting_coeffs import plot_material_properties, plot_conductivity_vs_temperature
from ..plottingcode.plotting_pin import plot_results_pin
from ..plottingcode.plotting_geometry import plot_pin, plot_pin_assembly, plot_plate, plot_plate_assembly
from code_architecture.helper_codes.material_properties.fuel_properties import calculate_k_fuel
from code_architecture.helper_codes.material_properties.clad_properties import calculate_k_clad
from code_architecture.helper_codes.material_properties.gap_properties import calculate_h_gap_vector
import numpy as np
import os
import sys

# Add root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

def generate_plots(th_system, output_dir=None):
    """Generate all plots for the thermal-hydraulic analysis.

    This function creates a comprehensive set of plots visualizing the thermal-hydraulic
    analysis results, including temperature distributions, material properties, and
    geometry visualizations.

    Args:
        th_system: THSystem object containing all thermal-hydraulic data
        output_dir (str, optional): Directory to save the plots. If None,
            uses the default output directory from th_system. Defaults to None.

    Generated plots include:
    For both pin and plate assemblies:
        - Temperature profiles (coolant, cladding, fuel)
        - Material properties (thermal conductivities)
        - Conductivity vs temperature relationships
        - Assembly geometry visualizations

    Additional plots for pin assemblies:
        - Gap heat transfer coefficient vs power
        - MDNBR distribution
        - Radial temperature profiles
    """
    # Set up output directory
    if output_dir is None:
        # Use the ThermalHydraulics directory as root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        output_dir = os.path.join(root_dir, 'ThermalHydraulics', th_system.thermal_hydraulics.outputs_folder)

    # Create output directories
    plots_dir = os.path.join(output_dir, 'TH_plots')
    geometry_plots_dir = os.path.join(output_dir, 'geometry_plots')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(geometry_plots_dir, exist_ok=True)

    # Set matplotlib to use the correct output directory
    import matplotlib.pyplot as plt
    plt.rcParams['savefig.directory'] = plots_dir

    # Get data from thermal state
    if th_system.thermal_hydraulics.assembly_type == 'Plate':
        T_clad_y = calculate_cladding_temperature_profile(
            th_system.thermal_state.Q_dot_z,
            th_system.thermal_state.T_clad_in_z,
            th_system.thermal_state.T_clad_out_z
        )
        plot_results_plate(
            th_system.thermal_state.Q_dot_z,
            th_system.thermal_state.T_coolant_z,
            th_system.thermal_state.T_clad_out_z,
            th_system.thermal_state.T_clad_middle_z,
            th_system.thermal_state.T_clad_in_z,
            th_system.thermal_state.T_fuel_y,
            T_clad_y,
            output_dir=plots_dir
        )
        plot_material_properties(
            th_system.z,
            th_system.thermal_state.k_fuel_bulk,
            th_system.thermal_state.k_clad_out,
            th_system.thermal_state.k_clad_mid,
            th_system.thermal_state.k_clad_in,
            th_system.thermal_state.h_coolant,
            h_gap=None,
            output_dir=plots_dir
        )
        plot_conductivity_vs_temperature(
            lambda T: calculate_k_fuel(th_system, T),
            lambda T: calculate_k_clad(th_system, T),
            calculate_h_gap_vector=None,
            output_dir=plots_dir
        )
        # Plot plate geometry
        plot_plate(geometry_plots_dir)
        plot_plate_assembly(geometry_plots_dir)
    else:  # Pin assembly
        # Create radial mesh for pin
        r_fuel_mesh = np.linspace(0, th_system.pin_geometry.r_fuel, th_system.thermal_hydraulics.n_radial)

        T_clad_y = calculate_cladding_temperature_profile(
            th_system.thermal_state.Q_dot_z,
            th_system.thermal_state.T_clad_in_z,
            th_system.thermal_state.T_clad_out_z
        )
        plot_results_pin(
            th_system.thermal_state.Q_dot_z,
            th_system.thermal_state.T_coolant_z,
            th_system.thermal_state.T_clad_out_z,
            th_system.thermal_state.T_clad_middle_z,
            th_system.thermal_state.T_clad_in_z,
            th_system.thermal_state.T_fuel_surface_z,
            th_system.thermal_state.T_fuel_centerline_z,
            th_system.thermal_state.T_fuel_y,
            r_fuel_mesh,
            th_system.thermal_state.MDNBR,
            output_dir=plots_dir
        )
        plot_material_properties(
            th_system.z,
            th_system.thermal_state.k_fuel_bulk,
            th_system.thermal_state.k_clad_out,
            th_system.thermal_state.k_clad_mid,
            th_system.thermal_state.k_clad_in,
            th_system.thermal_state.h_coolant,
            th_system.thermal_state.h_gap,
            output_dir=plots_dir
        )

        # Create a wrapper class that matches the THSystem interface needed for h_gap calculation
        class GapSystem:
            def __init__(self, gap_width):
                self.geometry = type('PinGeometry', (), {'gap_width': gap_width})()
                self.thermal_state = type('ThermalState', (), {'Q_dot_z': np.array([0])})()

        gap_system = GapSystem(th_system.pin_geometry.gap_width)

        def h_gap_function(power_kw_m):
            gap_system.thermal_state.Q_dot_z = np.array([power_kw_m * 1000])  # Convert kW/m to W/m
            return calculate_h_gap_vector(gap_system)[0]

        plot_conductivity_vs_temperature(
            lambda T: calculate_k_fuel(th_system, T),
            lambda T: calculate_k_clad(th_system, T),
            h_gap_function,
            output_dir=plots_dir
        )
        # Plot pin geometry
        plot_pin(geometry_plots_dir)
        plot_pin_assembly(geometry_plots_dir)

    # Close all figures to free memory
    plt.close('all')
