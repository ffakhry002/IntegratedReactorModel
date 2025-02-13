import numpy as np
from scipy import integrate

def calculate_temperature_points_plates(th_system):
    """Calculate temperature points for plate geometry.

    Args:
        th_system: THSystem object containing geometry, material information, and thermal state

    Returns:
        tuple: Arrays of temperatures at different points
            - T_coolant_z: Coolant temperatures
            - T_clad_out: Outer cladding temperatures
            - T_clad_middle: Middle cladding temperatures
            - T_clad_in: Inner cladding temperatures
            - T_fuel_surface: Fuel surface temperatures
            - T_fuel_centerline: Fuel centerline temperatures
            - T_fuel_avg: Average fuel temperatures
            - T_fuel_y: 2D array of fuel temperatures
            - y_fuel: Array of fuel mesh points
    """
    # Extract required properties from th_system
    Q_dot_z = th_system.thermal_state.Q_dot_z
    mass_flow_rate = th_system.thermal_state.mass_flow_rate
    specific_heat_capacity = th_system.thermal_state.specific_heat_capacity
    heat_transfer_coeff_coolant = th_system.thermal_state.h_coolant
    k_clad = th_system.thermal_state.k_clad_out  # Use outer cladding conductivity
    k_fuel = th_system.thermal_state.k_fuel

    integral_Q_dot_z = integrate.cumulative_trapezoid(Q_dot_z, th_system.z, initial=0)
    T_coolant_z = (1 / (mass_flow_rate * specific_heat_capacity)) * integral_Q_dot_z + th_system.thermal_hydraulics.T_inlet
    T_clad_out = T_coolant_z + Q_dot_z / (2 * th_system.plate_geometry.fuel_plate_width * heat_transfer_coeff_coolant)
    T_clad_middle = T_clad_out + Q_dot_z / (2*th_system.plate_geometry.fuel_plate_width) * (th_system.plate_geometry.clad_thickness/(2*k_clad))
    T_clad_in = T_clad_middle + Q_dot_z / (2*th_system.plate_geometry.fuel_plate_width) * (th_system.plate_geometry.clad_thickness/(2*k_clad))

    Q_triple_dot = Q_dot_z / (th_system.plate_geometry.fuel_meat_thickness * th_system.plate_geometry.fuel_meat_width)
    y_fuel = np.linspace(0, th_system.plate_geometry.fuel_meat_thickness / 2, th_system.thermal_hydraulics.n_radial)
    T_fuel_y = np.zeros((len(th_system.z), len(y_fuel)))

    for i in range(len(th_system.z)):
        for j in range(len(y_fuel)):
            T_fuel_y[i, j] = Q_triple_dot[i] / k_fuel[i,j] * ((th_system.plate_geometry.fuel_meat_thickness / 2) ** 2 - y_fuel[j] ** 2) + T_clad_in[i]

    T_fuel_centerline = T_fuel_y[:, 0]
    T_fuel_avg = np.mean(T_fuel_y, axis=1)
    T_fuel_surface = T_fuel_y[:, -1]

    return T_coolant_z, T_clad_out, T_clad_middle, T_clad_in, T_fuel_surface, T_fuel_centerline, T_fuel_avg, T_fuel_y, y_fuel
