import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import trapezoid

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from inputs import inputs

def initialize_globals(inputs_dict=None):
    """Initialize global variables for pin plotting from inputs.

    Args:
        inputs_dict (dict, optional): Custom inputs dictionary. If None, uses the global inputs.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    global pin_pitch, r_fuel, r_clad_inner, r_clad_outer, n_side_pins, n_guide_tubes
    global coolant_type, clad_type, fuel_type
    global core_power, num_assemblies, reactor_pressure, flow_rate, T_inlet, fuel_height, cos_curve_squeeze, assembly_type
    global z, output_folder

    # Pin Fuel Geometry
    pin_pitch = inputs_dict["pin_pitch"]
    r_fuel = inputs_dict["r_fuel"]
    r_clad_inner = inputs_dict["r_clad_inner"]
    r_clad_outer = inputs_dict["r_clad_outer"]
    n_side_pins = inputs_dict["n_side_pins"]
    n_guide_tubes = inputs_dict["n_guide_tubes"]

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

    z = np.linspace(-fuel_height/2, fuel_height/2, 1000)
    output_folder = inputs_dict['outputs_folder']

def setup_radial_grid(n_points=500):
    r_fuel_local = np.linspace(0, r_fuel, n_points)
    r_gap = np.linspace(r_fuel, r_clad_inner, n_points)
    r_clad = np.linspace(r_clad_inner, r_clad_outer, n_points)
    r_coolant = np.linspace(r_clad_outer, pin_pitch/2, n_points)
    r_total = np.concatenate((r_fuel_local, r_gap, r_clad, r_coolant))
    return r_total

def total_piecewise_temp(piecewise_temp):
    total_piecewise = np.zeros((piecewise_temp.shape[0], 2 * piecewise_temp.shape[1]))
    total_piecewise[:, :piecewise_temp.shape[1]] = np.fliplr(piecewise_temp)
    total_piecewise[:, piecewise_temp.shape[1]:] = piecewise_temp
    return total_piecewise

def piecewise_temperature(r, z_positions, T_coolant_z, T_clad_out_z, T_clad_in_z, T_fuel_surface_z, T_fuel_y, r_fuel_mesh):
    T = np.zeros((len(z_positions), len(r)))
    for i, z_pos in enumerate(z_positions):
        z_index = np.argmin(np.abs(z - z_pos))
        for j, r_val in enumerate(r):
            if r_val <= r_fuel:
                # Use the actual fuel temperature distribution
                fuel_index = np.argmin(np.abs(r_fuel_mesh - r_val))
                T[i, j] = T_fuel_y[z_index, fuel_index]
            elif r_fuel < r_val <= r_clad_inner:
                # Gap region - linear interpolation
                T[i, j] = T_fuel_surface_z[z_index] + (T_clad_in_z[z_index] - T_fuel_surface_z[z_index]) * (r_val - r_fuel) / (r_clad_inner - r_fuel)
            elif r_clad_inner < r_val <= r_clad_outer:
                # Cladding region - linear interpolation
                T[i, j] = T_clad_in_z[z_index] + (T_clad_out_z[z_index] - T_clad_in_z[z_index]) * (r_val - r_clad_inner) / (r_clad_outer - r_clad_inner)
            else:
                # Coolant region
                T[i, j] = T_coolant_z[z_index]
    return T

def plot_results_pin(Q_dot_z, T_coolant_z, T_clad_out_z, T_clad_middle_z, T_clad_in_z, T_fuel_surface_z, T_fuel_centerline_z, T_fuel_y, r_fuel_mesh, MDNBR, output_dir=None, element_power_mw=None, avg_element_power_mw=None, inputs_dict=None):
    initialize_globals(inputs_dict)
    if output_dir is None:
        # Use default directory if none provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(root_dir, output_folder, 'TH_plots')
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 40))

    # Axial Power Profile
    plt.subplot(8, 1, 1)
    plt.plot(z, Q_dot_z / 1000, label='Q_dot_z(z)')
    plt.xlabel('z (m)')
    plt.ylabel('Heat Generation Rate (kW/m)')
    plt.title('Axial Power Profile')
    plt.legend()
    plt.ylim(0, 1.2 * np.max(Q_dot_z / 1000))

    # Calculate total element power if not provided
    if element_power_mw is None:
        element_power_mw = trapezoid(Q_dot_z, z) / 1e6  # W to MW

    # Add text box with power information
    power_info = f"Element Power: {element_power_mw:.4f} MW"
    if avg_element_power_mw is not None:
        power_info += f"\nAvg Element Power: {avg_element_power_mw:.4f} MW"
        power_info += f"\nPower Ratio: {element_power_mw/avg_element_power_mw:.2f}"

    plt.text(0.02, 0.95, power_info, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Coolant Temperature
    plt.subplot(8, 1, 2)
    plt.plot(z, T_coolant_z, label='T_coolant(z)')
    plt.xlabel('z (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Coolant Temperature along Fuel Rod Length')
    plt.legend()
    plt.ylim(np.min(T_coolant_z) - 5, np.max(T_coolant_z) + 5)

    # Cladding and Gap Temperatures
    plt.subplot(8, 1, 3)
    plt.plot(z, T_clad_out_z, label='T_clad_out(z)')
    plt.plot(z, T_clad_middle_z, label='T_clad_middle(z)')
    plt.plot(z, T_clad_in_z, label='T_clad_in(z)')
    plt.plot(z, T_fuel_surface_z, label='T_fuel_surface(z)')
    plt.xlabel('z (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Cladding and Fuel Surface Temperatures along Fuel Rod Length')
    plt.legend()
    plt.ylim(min(np.min(T_clad_out_z), np.min(T_clad_in_z), np.min(T_fuel_surface_z)) - 5,
             max(np.max(T_clad_out_z), np.max(T_clad_in_z), np.max(T_fuel_surface_z)) + 5)

    # Fuel Temperatures
    plt.subplot(8, 1, 4)
    plt.plot(z, T_fuel_surface_z, label='T_fuel_surface(z)')
    plt.plot(z, T_fuel_centerline_z, label='T_fuel_centerline(z)')
    plt.xlabel('z (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Fuel Temperatures along Fuel Rod Length')
    plt.legend()
    plt.ylim(min(np.min(T_fuel_surface_z) - 5, np.min(T_fuel_centerline_z) - 5),
             max(np.max(T_fuel_surface_z) + 5, np.max(T_fuel_centerline_z) + 5))

    # Radial Temperature Profiles
    plt.subplot(8, 1, 5)
    r_total = np.linspace(0, pin_pitch/2, 1000)
    z_positions = [-fuel_height / 2, -fuel_height / (2*3), 0, fuel_height / (2*3), fuel_height / 2]
    for z_pos in z_positions:
        i = np.argmin(np.abs(z - z_pos))
        T = piecewise_temperature(r_total, [z_pos], T_coolant_z, T_clad_out_z, T_clad_in_z, T_fuel_surface_z, T_fuel_y, r_fuel_mesh)
        plt.plot(r_total * 100, T[0], label=f'T at z={z[i]:.2f} m')
    plt.xlabel('r (cm)')
    plt.ylabel('Temperature (K)')
    plt.title('Radial Temperature Profiles at Different z')
    plt.legend()
    plt.xlim(0, r_total[-1] * 100)

    # Fuel Temperature as a Function of Radial Position
    plt.subplot(8, 1, 6)
    for z_pos in z_positions:
        i = np.argmin(np.abs(z - z_pos))
        plt.plot(r_fuel_mesh * 100, T_fuel_y[i], label=f'T_fuel at z={z_pos:.2f} m')
    plt.xlabel('r (cm)')
    plt.ylabel('Temperature (K)')
    plt.title('Fuel Temperature as a Function of Radial Position for Different z')
    plt.legend()
    plt.xlim(0, r_fuel * 100)

    # Radial Temperature Profile at z=0
    plt.subplot(8, 1, 7)
    z_pos = 0
    i = np.argmin(np.abs(z - z_pos))
    T = piecewise_temperature(r_total, [z_pos], T_coolant_z, T_clad_out_z, T_clad_in_z, T_fuel_surface_z, T_fuel_y, r_fuel_mesh)

    plt.plot(r_total[r_total <= r_fuel] * 100, T[0][r_total <= r_fuel], 'r', label='Fuel')
    plt.plot(r_total[(r_total > r_fuel) & (r_total <= r_clad_inner)] * 100,
             T[0][(r_total > r_fuel) & (r_total <= r_clad_inner)], 'y', label='Gap')
    plt.plot(r_total[(r_total > r_clad_inner) & (r_total <= r_clad_outer)] * 100,
             T[0][(r_total > r_clad_inner) & (r_total <= r_clad_outer)], 'g', label='Cladding')
    plt.plot(r_total[r_total > r_clad_outer] * 100, T[0][r_total > r_clad_outer], 'b', label='Coolant')

    plt.xlabel('r (cm)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Radial Temperature Profile at z={z[i]:.2f} m')
    plt.legend()
    plt.xlim(0, r_total[-1] * 100)
    plt.ylim(np.min(T) - 50, np.max(T) + 50)

    # MDNBR
    plt.subplot(8, 1, 8)
    plt.plot(z, MDNBR)
    plt.xlabel('Axial Height (m)')
    plt.ylabel('MDNBR')
    plt.title('Minimum Departure from Nucleate Boiling Ratio (MDNBR) vs Axial Height')
    plt.grid(True)
    plt.axhline(y=2, color='r', linestyle='--', label='MDNBR = 2')
    plt.legend()
    min_mdnbr = np.min(MDNBR)
    plt.text(0.05, 0.95, f'Min DNBR: {min_mdnbr:.2f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.ylim(0, 50)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temperature_profiles.png'))
    plt.close()
