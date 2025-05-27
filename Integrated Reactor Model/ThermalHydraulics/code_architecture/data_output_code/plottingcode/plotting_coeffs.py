# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from utils.base_inputs import inputs
from ThermalHydraulics.code_architecture.helper_codes.material_properties.gap_properties import calculate_h_gap_vector

# Add this class to simulate the th_system structure needed by calculate_h_gap_vector
class MockTHSystem:
    def __init__(self, Q_dot_z, gap_width):
        self.thermal_state = type('obj', (object,), {'Q_dot_z': Q_dot_z})
        self.geometry = type('obj', (object,), {'gap_width': gap_width})

def initialize_globals(inputs_dict=None):
    """Initialize global variables for coefficient plotting from inputs.

    Args:
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    global pin_pitch, r_fuel, r_clad_inner, r_clad_outer, n_side_pins, n_guide_tubes
    global fuel_meat_width, fuel_plate_width, fuel_plate_pitch, fuel_meat_thickness, plates_per_assembly, clad_structure_width, clad_thickness
    global coolant_type, clad_type, fuel_type, fuel_plate_thickness
    global core_power, num_assemblies, reactor_pressure, flow_rate, T_inlet, fuel_height, cos_curve_squeeze, assembly_type
    global z, output_folder

    # Pin Fuel Geometry
    pin_pitch = inputs_dict["pin_pitch"]
    r_fuel = inputs_dict["r_fuel"]
    r_clad_inner = inputs_dict["r_clad_inner"]
    r_clad_outer = inputs_dict["r_clad_outer"]
    n_side_pins = inputs_dict["n_side_pins"]
    n_guide_tubes = inputs_dict["n_guide_tubes"]

    # Plate Fuel Geometry
    fuel_meat_width = inputs_dict["fuel_meat_width"]
    fuel_plate_width = inputs_dict["fuel_plate_width"]
    fuel_plate_pitch = inputs_dict["fuel_plate_pitch"]
    fuel_meat_thickness = inputs_dict["fuel_meat_thickness"]
    clad_thickness = inputs_dict["clad_thickness"]
    plates_per_assembly = inputs_dict["plates_per_assembly"]
    clad_structure_width = inputs_dict["clad_structure_width"]

    # Material profile
    coolant_type = inputs_dict["coolant_type"]
    clad_type = inputs_dict["clad_type"]
    fuel_type = inputs_dict["fuel_type"]

    # Reactor Parameters
    core_power = inputs_dict["core_power"]
    num_assemblies = inputs_dict["num_assemblies"]
    reactor_pressure = inputs_dict["reactor_pressure"]
    flow_rate = inputs_dict["flow_rate"]
    T_inlet = inputs_dict["T_inlet"]
    fuel_height = inputs_dict["fuel_height"]
    cos_curve_squeeze = inputs_dict["cos_curve_squeeze"]
    assembly_type = inputs_dict["assembly_type"]
    fuel_plate_thickness = fuel_meat_thickness + 2*clad_thickness

    z = np.linspace(-fuel_height/2, fuel_height/2, 1000)
    output_folder = inputs_dict['outputs_folder']

def plot_material_properties(z, k_fuel, k_clad_out, k_clad_mid, k_clad_in, heat_transfer_coeff_coolant=None, h_gap=None, output_dir=None, inputs_dict=None):
    initialize_globals(inputs_dict)
    if output_dir is None:
        # Use default directory if none provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(root_dir, output_folder, 'TH_plots')
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # Plot 1: Fuel and Cladding Thermal Conductivity
    ax1.plot(z, k_fuel, label=f'{fuel_type} Fuel Bulk', color='r')
    ax1.plot(z, k_clad_out, label=f'{clad_type} Clad Outer', color='b')
    ax1.plot(z, k_clad_mid, label=f'{clad_type} Clad Middle', color='g')
    ax1.plot(z, k_clad_in, label=f'{clad_type} Clad Inner', color='c')

    ax1.set_xlabel('Axial Position (m)')
    ax1.set_ylabel('Thermal Conductivity (W/m-K)')
    ax1.set_title('Fuel and Cladding Thermal Conductivity along Fuel Length')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Heat Transfer Coefficients
    ax2.plot(z, heat_transfer_coeff_coolant, label=f'{coolant_type} Coolant HTC', color='m')

    if h_gap is not None:
        ax2.plot(z, h_gap, label='Gap HTC', color='y')

    ax2.set_xlabel('Axial Position (m)')
    ax2.set_ylabel('Heat Transfer Coefficient (W/m²-K)')
    ax2.set_title('Heat Transfer Coefficients along Fuel Length')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'material_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_conductivity_vs_temperature(calculate_k_fuel, calculate_k_clad, calculate_h_gap_vector, output_dir=None, inputs_dict=None):
    initialize_globals(inputs_dict)
    if output_dir is None:
        # Use default directory if none provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(root_dir, output_folder, 'TH_plots')
        os.makedirs(output_dir, exist_ok=True)

    # Create a figure for fuel and cladding conductivity
    plt.figure(figsize=(12, 8))

    # Plot fuel and clad conductivity
    temperatures = np.linspace(100, 2500, 1200)  # K
    k_fuel = np.vectorize(calculate_k_fuel)(temperatures)
    k_clad = np.vectorize(calculate_k_clad)(temperatures)

    plt.plot(temperatures, k_fuel, label='Fuel', color='r', linewidth=2)
    plt.plot(temperatures, k_clad, label='Cladding', color='b', linewidth=2)
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Thermal Conductivity (W/m-K)', fontsize=14)
    plt.title('Thermal Conductivity vs Temperature for Fuel and Cladding', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Conductivity_vs_Temperature.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # If gap heat transfer is included, create a separate plot using our specialized function
    if calculate_h_gap_vector:
        plot_gap_heat_transfer_coefficient(output_dir, inputs_dict)

def plot_gap_heat_transfer_coefficient(output_dir=None, inputs_dict=None):
    """
    Plot the gap heat transfer coefficient (h_gap) as a function of linear power
    for the current gap width, with red dots marking the interpolated points.

    Args:
        output_dir: Directory to save the plot file. If None, uses default.
    """
    initialize_globals(inputs_dict)
    if output_dir is None:
        # Use default directory if none provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(root_dir, output_folder, 'TH_plots')
        os.makedirs(output_dir, exist_ok=True)

    # Get the gap width from inputs (for pin design)
    gap_width = r_clad_inner - r_fuel

    # Create figure
    plt.figure(figsize=(12, 8))

    # Define specific linear power points where we want to show the interpolated values
    # Use the same points as in the data tables (convert W/cm to W/m)
    key_powers_W_m = np.array([170, 330, 380, 450, 500]) * 100

    # Create a smooth curve with more points for the continuous line
    linear_power_W_m = np.linspace(1000, 50000, 500)  # 1 to 50 kW/m

    # Plot h_gap for current gap width - continuous line
    mock_system = MockTHSystem(linear_power_W_m, gap_width)
    h_gap_values = calculate_h_gap_vector(mock_system)
    plt.plot(linear_power_W_m/1000, h_gap_values, 'g-', linewidth=2,
             label=f'Gap width: {gap_width*1e6:.1f} µm')

    # Calculate h_gap values at the specific key points
    mock_system_key_points = MockTHSystem(key_powers_W_m, gap_width)
    h_gap_key_points = calculate_h_gap_vector(mock_system_key_points)

    # Plot the interpolated points as red dots
    plt.plot(key_powers_W_m/1000, h_gap_key_points, 'ro', markersize=8,
             label='Interpolated data points')

    plt.xlabel('Linear Power (kW/m)', fontsize=14)
    plt.ylabel('Gap Heat Transfer Coefficient (W/m²-K)', fontsize=14)
    plt.title('Gap Heat Transfer Coefficient vs. Linear Power', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'Gap_HTC_vs_Power.png'), dpi=300, bbox_inches='tight')
    plt.close()
