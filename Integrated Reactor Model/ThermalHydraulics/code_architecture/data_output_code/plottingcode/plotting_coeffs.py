import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from inputs import inputs

def initialize_globals():
    global pin_pitch, r_fuel, r_clad_inner, r_clad_outer, n_side_pins, n_guide_tubes
    global fuel_meat_width, fuel_plate_width, fuel_plate_pitch, fuel_meat_thickness, plates_per_assembly, clad_structure_width, clad_thickness
    global coolant_type, clad_type, fuel_type, fuel_plate_thickness
    global core_power, num_assemblies, reactor_pressure, flow_rate, T_inlet, fuel_height, cos_curve_squeeze, assembly_type
    global z, output_folder

    # Pin Fuel Geometry
    pin_pitch = inputs["pin_pitch"]
    r_fuel = inputs["r_fuel"]
    r_clad_inner = inputs["r_clad_inner"]
    r_clad_outer = inputs["r_clad_outer"]
    n_side_pins = inputs["n_side_pins"]
    n_guide_tubes = inputs["n_guide_tubes"]

    # Plate Fuel Geometry
    fuel_meat_width = inputs["fuel_meat_width"]
    fuel_plate_width = inputs["fuel_plate_width"]
    fuel_plate_pitch = inputs["fuel_plate_pitch"]
    fuel_meat_thickness = inputs["fuel_meat_thickness"]
    clad_thickness = inputs["clad_thickness"]
    plates_per_assembly = inputs["plates_per_assembly"]
    clad_structure_width = inputs["clad_structure_width"]

    # Material profile
    coolant_type = inputs["coolant_type"]
    clad_type = inputs["clad_type"]
    fuel_type = inputs["fuel_type"]

    # Reactor Parameters
    core_power = inputs["core_power"]
    num_assemblies = inputs["num_assemblies"]
    reactor_pressure = inputs["reactor_pressure"]
    flow_rate = inputs["flow_rate"]
    T_inlet = inputs["T_inlet"]
    fuel_height = inputs["fuel_height"]
    cos_curve_squeeze = inputs["cos_curve_squeeze"]
    assembly_type = inputs["assembly_type"]
    fuel_plate_thickness = fuel_meat_thickness + 2*clad_thickness

    z = np.linspace(-fuel_height/2, fuel_height/2, 1000)
    output_folder = inputs['outputs_folder']

def plot_material_properties(z, k_fuel, k_clad_out, k_clad_mid, k_clad_in, heat_transfer_coeff_coolant=None, h_gap=None, output_dir=None):
    initialize_globals()
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
    plt.savefig(os.path.join(output_dir, f'material_properties.png'))
    plt.close()

def plot_conductivity_vs_temperature(calculate_k_fuel, calculate_k_clad, calculate_h_gap_vector, output_dir=None):
    initialize_globals()
    if output_dir is None:
        # Use default directory if none provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(root_dir, output_folder, 'TH_plots')
        os.makedirs(output_dir, exist_ok=True)

    # Create a figure with two subplots
    num = 2 if calculate_h_gap_vector else 1
    fig, axes = plt.subplots(num, 1, figsize=(12, 16))

    # Subplot 1: Fuel and Clad Conductivity
    temperatures = np.linspace(100, 2500, 1200)  # K
    k_fuel = np.vectorize(calculate_k_fuel)(temperatures)
    k_clad = np.vectorize(calculate_k_clad)(temperatures)

    ax1 = axes[0] if num == 2 else axes
    ax1.plot(temperatures, k_fuel, label='Fuel', color='r')
    ax1.plot(temperatures, k_clad, label='Cladding', color='b')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Thermal Conductivity (W/m-K)')
    ax1.set_title('Thermal Conductivity vs Temperature for Fuel and Cladding')
    ax1.legend()
    ax1.grid(True)

    if calculate_h_gap_vector:
        # Subplot 2: h_gap vs linear power
        linear_power_kw_m = np.linspace(0, 50, 1000)  # kW/m
        h_gap = calculate_h_gap_vector(linear_power_kw_m * 1000)  # Convert to W/m for the function

        ax2 = axes[1]
        ax2.plot(linear_power_kw_m, h_gap, color='g')
        ax2.set_xlabel('Linear Power (kW/m)')
        ax2.set_ylabel('h_gap (W/m^2-K)')
        ax2.set_title('h_gap vs Linear Power')
        ax2.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Conductivity_Coefficients.png'))
    plt.close()
