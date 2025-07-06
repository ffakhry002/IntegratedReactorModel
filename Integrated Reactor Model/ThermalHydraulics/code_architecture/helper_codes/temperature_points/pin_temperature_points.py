import numpy as np
from scipy import integrate

def calculate_temperature_points_pins(th_system):
    """Calculate temperature points for pin geometry.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing geometry, material information, and thermal state

    Returns
    -------
    tuple
        Arrays of temperatures at different points
            - T_coolant_z: Coolant temperatures
            - T_clad_out: Outer cladding temperatures
            - T_clad_middle: Middle cladding temperatures
            - T_clad_in: Inner cladding temperatures
            - T_fuel_surface: Fuel surface temperatures
            - T_fuel_centerline: Fuel centerline temperatures
            - T_fuel_avg: Average fuel temperatures
            - T_fuel_y: 2D array of fuel temperatures
            - r_fuel_mesh: Radial mesh points
    """
    # Extract required properties from th_system
    Q_dot_z = th_system.thermal_state.Q_dot_z
    mass_flow_rate = th_system.thermal_state.mass_flow_rate
    specific_heat_capacity = th_system.thermal_state.specific_heat_capacity
    heat_transfer_coeff_coolant = th_system.thermal_state.h_coolant
    k_clad_out = th_system.thermal_state.k_clad_out
    k_clad_mid = th_system.thermal_state.k_clad_mid
    k_clad_in = th_system.thermal_state.k_clad_in
    k_fuel = th_system.thermal_state.k_fuel
    h_gap = th_system.thermal_state.h_gap

    integral_Q_dot_z = integrate.cumulative_trapezoid(Q_dot_z, th_system.z, initial=0)
    T_coolant_z = (1 / (mass_flow_rate * specific_heat_capacity)) * integral_Q_dot_z + th_system.thermal_hydraulics.T_inlet
    T_clad_out = T_coolant_z + Q_dot_z / (2 * np.pi * th_system.pin_geometry.r_clad_outer * heat_transfer_coeff_coolant)
    r_clad_middle = (th_system.pin_geometry.r_clad_inner + th_system.pin_geometry.r_clad_outer) / 2
    T_clad_middle = T_clad_out + Q_dot_z / (2 * np.pi * k_clad_out) * np.log(th_system.pin_geometry.r_clad_outer / r_clad_middle)
    T_clad_in = T_clad_middle + Q_dot_z / (2 * np.pi * k_clad_mid) * np.log(r_clad_middle / th_system.pin_geometry.r_clad_inner)
    T_fuel_surface = T_clad_in + Q_dot_z / (2 * np.pi * th_system.pin_geometry.r_fuel * h_gap)

    q_v = Q_dot_z / (np.pi * th_system.pin_geometry.r_fuel**2)
    r_fuel_mesh = np.linspace(0, th_system.pin_geometry.r_fuel, th_system.thermal_hydraulics.n_radial)
    T_fuel_y = np.zeros((len(th_system.z), len(r_fuel_mesh)))
    for i in range(len(th_system.z)):
        T_fuel_y[i] = q_v[i] / (4 * k_fuel[i]) * th_system.pin_geometry.r_fuel**2 * (1 - (r_fuel_mesh / th_system.pin_geometry.r_fuel)**2) + T_fuel_surface[i]

    T_fuel_centerline = T_fuel_y[:, 0]
    T_fuel_avg = np.mean(T_fuel_y, axis=1)

    return T_coolant_z, T_clad_out, T_clad_middle, T_clad_in, T_fuel_surface, T_fuel_centerline, T_fuel_avg, T_fuel_y, r_fuel_mesh
