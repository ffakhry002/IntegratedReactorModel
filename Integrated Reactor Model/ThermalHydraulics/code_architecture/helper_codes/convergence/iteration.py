import numpy as np
from ..material_properties.coolant_properties import (
    get_coolant_properties,
    calculate_heat_transfer_coeff_coolant,
    calculate_mass_flow_rate
)

def single_iteration(th_system):
    """Perform a single iteration of temperature calculations using current thermal state.

    This function updates the thermal state with new coolant properties, heat transfer
    coefficients, and calculates new temperature distributions based on the current state.

    Args:
        th_system: THSystem object containing geometry, material information, and current thermal state

    Returns:
        tuple: Contains the following arrays:
            - T_coolant_z: Coolant temperatures
            - T_clad_out_z: Outer cladding temperatures
            - T_clad_middle_z: Middle cladding temperatures
            - T_clad_in_z: Inner cladding temperatures
            - T_fuel_surface_z: Fuel surface temperatures
            - T_fuel_centerline_z: Fuel centerline temperatures
            - T_fuel_avg_z: Average fuel temperatures
            - T_fuel_y: 2D array of fuel temperatures
            - y_fuel: Radial mesh points
            - Q_dot_z: Heat generation rates
            - mass_flow_rate: Mass flow rate
            - heat_transfer_coeff_coolant: Coolant heat transfer coefficient
    """
    # Get coolant properties from the main system
    coolant_density, specific_heat_capacity, thermal_conductivity, viscosity = get_coolant_properties(
        th_system, th_system.thermal_state.T_coolant_z)

    # Calculate heat transfer coefficients
    heat_transfer_coeff_coolant = calculate_heat_transfer_coeff_coolant(th_system)

    # Calculate power and flow
    Q_dot_z = th_system.calculate_Q_dot_z()
    mass_flow_rate = calculate_mass_flow_rate(th_system)

    # Update thermal state with new properties
    th_system.thermal_state.coolant_density = coolant_density
    th_system.thermal_state.specific_heat_capacity = specific_heat_capacity
    th_system.thermal_state.thermal_conductivity = thermal_conductivity
    th_system.thermal_state.viscosity = viscosity
    th_system.thermal_state.h_coolant = heat_transfer_coeff_coolant
    th_system.thermal_state.Q_dot_z = Q_dot_z
    th_system.thermal_state.mass_flow_rate = mass_flow_rate

    # Calculate temperatures using the main system's methods
    if th_system.thermal_hydraulics.assembly_type == 'Pin':
        results = th_system.calculate_temperature_points_pins()
    else:
        results = th_system.calculate_temperature_points_plates()

    return (*results, Q_dot_z, mass_flow_rate, heat_transfer_coeff_coolant)
