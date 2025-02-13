import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from inputs import inputs

def initialize_globals():
    """Initialize global variables for plate plotting from inputs.

    Sets up global variables for geometry dimensions, mesh points, and output folder
    from the inputs configuration. Creates arrays for spatial discretization in
    both axial and radial directions.
    """
    global pin_pitch, r_fuel, r_clad_inner, r_clad_outer, n_side_pins, n_guide_tubes
    global fuel_meat_width, fuel_plate_width, fuel_plate_pitch, fuel_meat_thickness, plates_per_assembly, clad_structure_width, clad_thickness
    global coolant_type, clad_type, fuel_type, fuel_plate_thickness
    global core_power, num_assemblies, reactor_pressure, flow_rate, T_inlet, fuel_height, cos_curve_squeeze, assembly_type
    global z, y_fuel, y_clad, y_water, y_total, y_total_mirror, output_folder

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
    y_fuel = np.linspace(0, fuel_plate_thickness / 2, 500)
    y_clad = np.linspace(fuel_plate_thickness / 2, fuel_plate_thickness / 2 + clad_thickness, 500)
    y_water = np.linspace(fuel_plate_thickness / 2 + clad_thickness, fuel_plate_thickness / 2 + clad_thickness + (fuel_plate_pitch - fuel_plate_thickness) / 2, 500)
    y_total = np.concatenate((y_fuel, y_clad, y_water))

    y_total_mirror = np.zeros(2 * len(y_total))
    y_total_mirror[:len(y_total)] = -y_total[::-1]
    y_total_mirror[len(y_total):] = y_total

    output_folder = inputs['outputs_folder']


def piecewise_temperature(y, z_positions, T_coolant_z, T_clad_y, T_fuel_y):
    T = np.zeros_like(y)
    for z_pos in z_positions:
        i = np.argmin(np.abs(z - z_pos))
        for j in range(len(y)):
            if y[j] < fuel_plate_thickness / 2:
                fuel_index = int(y[j] / (fuel_plate_thickness / 2) * (len(y_fuel) - 1))
                T[j] = T_fuel_y[i, fuel_index]
            elif fuel_plate_thickness / 2 <= y[j] < fuel_plate_thickness / 2 + clad_thickness:
                clad_index = int((y[j] - fuel_plate_thickness / 2) / clad_thickness * (len(y_clad) - 1))
                T[j] = T_clad_y[i, clad_index]
            else:
                T[j] = T_coolant_z[i]
    return T

def total_piecewise_temp(piecewise_temp):
    """Create a mirrored temperature profile.

    Args:
        piecewise_temp (np.array): Original temperature profile array

    Returns:
        np.array: Mirrored temperature profile with twice the length of input,
                 where the first half is the reversed input and second half is the input
    """
    total_piecewise = np.zeros(2 * len(piecewise_temp))
    total_piecewise[:len(piecewise_temp)] = piecewise_temp[::-1]
    total_piecewise[len(piecewise_temp):] = piecewise_temp
    return total_piecewise

def generate_temperature_profiles(T_coolant_z, T_clad_y, T_fuel_y):
    temp_profiles = []
    for z_pos in z:
        piecewise_temp = piecewise_temperature(y_total, [z_pos], T_coolant_z, T_clad_y, T_fuel_y)
        mirrored_piecewise = total_piecewise_temp(piecewise_temp)
        temp_profiles.append((z_pos, y_total_mirror, mirrored_piecewise))
    return temp_profiles

def calculate_cladding_temperature_profile(Q_dot_z, T_clad_in, T_clad_out):
    """Calculate temperature profile across cladding thickness.

    Args:
        Q_dot_z (np.array): Heat generation rate along z-axis
        T_clad_in (np.array): Inner cladding surface temperatures
        T_clad_out (np.array): Outer cladding surface temperatures

    Returns:
        np.array: 2D array of cladding temperatures (z, y)
    """
    initialize_globals()
    T_clad_y = np.zeros((len(z), len(y_clad)))
    for i in range(len(z)):
        T_clad_y[i, :] = T_clad_in[i] - (T_clad_in[i] - T_clad_out[i]) * ((y_clad - fuel_plate_thickness / 2) / clad_thickness)
    return T_clad_y

def plot_results_plate(Q_dot_z, T_coolant_z, T_clad_out, T_clad_middle,T_clad_in, T_fuel_y, T_clad_y, output_dir=None):
    initialize_globals()
    if output_dir is None:
        # Use default directory if none provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(root_dir, output_folder, 'TH_plots')
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 28))

    plt.subplot(7, 1, 1)
    plt.plot(z, Q_dot_z / (1000), label='Q_dot_z(z)')
    plt.xlabel('z (m)')
    plt.ylabel('Heat Generation Rate (kW/m) Per Plate')
    plt.title('Axial Power Profile')
    plt.legend()
    plt.ylim(0, 1.2 * np.max(Q_dot_z / (1000)))  # Adjust y-axis limits

    plt.subplot(7, 1, 2)
    plt.plot(z, T_coolant_z, label='T_coolant(z)')
    plt.xlabel('z (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Coolant Temperature along Assembly Length')
    plt.legend()
    plt.ylim(np.min(T_coolant_z) - 5, np.max(T_coolant_z) + 5)  # Adjust y-axis limits

    plt.subplot(7, 1, 3)
    plt.plot(z, T_clad_out, label='T_clad_out(z)')
    plt.plot(z, T_clad_middle, label='T_clad_middle(z)')
    plt.plot(z, T_clad_in, label='T_clad_in(z)')
    plt.xlabel('z (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Cladding Temperatures along Fuel Length (Inner cladding and outer)')
    plt.legend()
    plt.ylim(min(np.min(T_clad_out) - 5, np.min(T_clad_in) - 5), max(np.max(T_clad_out) + 5, np.max(T_clad_in) + 5))

    plt.subplot(7, 1, 4)
    z_positions = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]  # z positions in meters
    for z_pos in z_positions:
        i = np.argmin(np.abs(z - z_pos))  # Find the index closest to the desired z position
        plt.plot(y_fuel * 100, T_fuel_y[i, :], label=f'T_fuel_y at z={z[i] * 100:.0f} cm')
    plt.xlabel('x (cm)')
    plt.ylabel('Temperature (K)')
    plt.title('Fuel Temperature across y at Different z')
    plt.legend()
    plt.xlim(0, fuel_plate_thickness / 2 * 100)
    plt.ylim(np.min(T_fuel_y) - 5, np.max(T_fuel_y) + 5)

    plt.subplot(7, 1, 5)
    for z_pos in z_positions:
        i = np.argmin(np.abs(z - z_pos))  # Find the index closest to the desired z position
        plt.plot(y_clad * 100, T_clad_y[i, :], label=f'T_clad_y at z={z[i] * 100:.0f} cm')
    plt.xlabel('x (cm)')
    plt.ylabel('Temperature (K)')
    plt.title('Cladding Temperature y at Different z')
    plt.legend()
    plt.xlim(fuel_plate_thickness / 2 * 100, (fuel_plate_thickness / 2 + clad_thickness) * 100)
    plt.ylim(np.min(T_clad_y) - 5, np.max(T_clad_y) + 5)

    z_positions = [0.0]
    plt.subplot(7, 1, 6)
    for z_pos in z_positions:
        i = np.argmin(np.abs(z - z_pos))  # Find the index closest to the desired z position
        piecewise_temp = piecewise_temperature(y_total, [z_pos], T_coolant_z, T_clad_y, T_fuel_y)
        plt.plot(y_fuel * 100, piecewise_temp[:len(y_fuel)], 'r', label='Fuel')
        plt.plot(y_clad * 100, piecewise_temp[len(y_fuel):len(y_fuel) + len(y_clad)], 'g', label='Cladding')
        plt.plot(y_water * 100, piecewise_temp[len(y_fuel) + len(y_clad):], 'b', label='Water')
        plt.xlabel('x (cm)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Piecewise Temperature Profile at z={z[i] * 100:.0f} cm')
        plt.legend()
        plt.xlim(0, y_total[-1] * 100)
        plt.ylim(np.min(piecewise_temp) - 5, np.max(piecewise_temp) + 5)

    plt.subplot(7, 1, 7)
    for z_pos in z_positions:
        i = np.argmin(np.abs(z - z_pos))  # Find the index closest to the desired z position
        piecewise_temp = piecewise_temperature(y_total, [z_pos], T_coolant_z, T_clad_y, T_fuel_y)
        mirrored_piecewise = total_piecewise_temp(piecewise_temp)

        water_left_boundary = -fuel_plate_pitch / 2
        water_right_boundary = -fuel_plate_thickness / 2 - clad_thickness
        clad_left_boundary = water_right_boundary
        clad_right_boundary = -fuel_plate_thickness / 2
        fuel_left_boundary = clad_right_boundary
        fuel_right_boundary = 0
        clad_left_boundary_2 = fuel_plate_thickness / 2
        clad_right_boundary_2 = clad_left_boundary_2 + clad_thickness
        water_left_boundary_2 = clad_right_boundary_2
        water_right_boundary_2 = fuel_plate_pitch / 2

        water_left_index = np.argmin(np.abs(y_total_mirror - water_left_boundary))
        water_right_index = np.argmin(np.abs(y_total_mirror - water_right_boundary))
        clad_left_index = water_right_index
        clad_right_index = np.argmin(np.abs(y_total_mirror - clad_right_boundary))
        fuel_left_index = clad_right_index
        fuel_right_index = np.argmin(np.abs(y_total_mirror - fuel_right_boundary))
        clad_left_index_2 = np.argmin(np.abs(y_total_mirror - clad_left_boundary_2))
        clad_right_index_2 = np.argmin(np.abs(y_total_mirror - clad_right_boundary_2))
        water_left_index_2 = clad_right_index_2
        water_right_index_2 = len(y_total_mirror) - 1

        plt.plot(y_total_mirror[water_left_index:water_right_index + 1] * 100, mirrored_piecewise[water_left_index:water_right_index + 1], 'b', label='Water')
        plt.plot(y_total_mirror[clad_left_index:clad_right_index + 1] * 100, mirrored_piecewise[clad_left_index:clad_right_index + 1], 'g', label='Cladding')
        plt.plot(y_total_mirror[fuel_left_index:fuel_right_index + 1] * 100, mirrored_piecewise[fuel_left_index:fuel_right_index + 1], 'r', label='Fuel')
        plt.plot(y_total_mirror[fuel_right_index:clad_left_index_2 + 1] * 100, mirrored_piecewise[fuel_right_index:clad_left_index_2 + 1], 'r')
        plt.plot(y_total_mirror[clad_left_index_2:clad_right_index_2 + 1] * 100, mirrored_piecewise[clad_left_index_2:clad_right_index_2 + 1], 'g')
        plt.plot(y_total_mirror[water_left_index_2:water_right_index_2 + 1] * 100, mirrored_piecewise[water_left_index_2:water_right_index_2 + 1], 'b')

        plt.xlabel('x (cm)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Mirrored Piecewise Temperature Profile at z={z[i] * 100:.0f} cm')
        plt.xlim(y_total_mirror[0] * 100, y_total_mirror[-1] * 100)
        plt.ylim(np.min(mirrored_piecewise) - 5, np.max(mirrored_piecewise) + 5)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temperature_profiles.png'))
