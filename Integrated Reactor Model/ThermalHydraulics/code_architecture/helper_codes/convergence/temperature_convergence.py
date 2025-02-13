import numpy as np
from .coolant_convergence import converge_coolant
from .clad_convergence import converge_cladding
from .fuel_convergence import converge_fuel
from ..material_properties.coolant_properties import (
    get_coolant_properties,
    calculate_heat_transfer_coeff_coolant,
    calculate_mass_flow_rate
)

def update_final_thermal_state(th_system):
    """Update the final thermal state after convergence.

    This function updates various thermal state parameters after the convergence process,
    including temperatures, thermal conductivities, coolant properties, heat transfer
    coefficients, and power distribution.

    Args:
        th_system: THSystem object containing geometry, material information, and thermal state
    """
    # Update temperatures
    th_system.thermal_state.T_fuel_centerline_z = th_system.thermal_state.T_fuel_y[:, 0]
    th_system.thermal_state.T_fuel_surface_z = th_system.thermal_state.T_fuel_y[:, -1]

    # Update fuel thermal conductivities
    th_system.thermal_state.k_fuel_centerline = th_system.thermal_state.k_fuel[:, 0]
    th_system.thermal_state.k_fuel_bulk = np.mean(th_system.thermal_state.k_fuel, axis=1)

    if th_system.thermal_hydraulics.assembly_type == 'Pin':
        th_system.thermal_state.T_gap_z = ((th_system.thermal_state.T_clad_in_z +
                                        th_system.thermal_state.T_fuel_surface_z) / 2)

    # Update coolant properties
    (th_system.thermal_state.coolant_density,
     th_system.thermal_state.specific_heat_capacity,
     th_system.thermal_state.thermal_conductivity,
     th_system.thermal_state.viscosity) = get_coolant_properties(th_system, th_system.thermal_state.T_coolant_z)

    # Update heat transfer coefficients
    th_system.thermal_state.h_coolant = calculate_heat_transfer_coeff_coolant(th_system)

    # Update mass flow rate
    th_system.thermal_state.mass_flow_rate = calculate_mass_flow_rate(th_system)

    # Update power distribution
    th_system.thermal_state.Q_dot_z = th_system.calculate_Q_dot_z()

def converge_temperatures(th_system, tolerance=0.001, max_iterations=100):
    """Perform complete temperature convergence for all components.

    This function manages the overall convergence process by:
    1. Initializing the thermal state
    2. Converging coolant temperatures
    3. Converging cladding temperatures and conductivities
    4. Converging fuel temperatures and conductivities
    5. Updating the final thermal state
    6. Calculating critical heat flux and MDNBR

    Args:
        th_system: THSystem object containing geometry and material information
        tolerance (float, optional): Convergence tolerance. Defaults to 0.001.
        max_iterations (int, optional): Maximum iterations for each convergence step. Defaults to 100.

    Returns:
        ThermalState: Converged thermal state object
    """
    # Initialize thermal state
    initial_temp = 310  # K
    th_system.thermal_state.initialize_state(
        initial_temp,
        th_system.thermal_hydraulics.n_axial,
        th_system.thermal_hydraulics.n_radial,
        th_system
    )

    # Converge temperatures
    th_system.thermal_state.T_coolant_z, _ = converge_coolant(th_system, tolerance, max_iterations)

    (th_system.thermal_state.T_clad_out_z, th_system.thermal_state.T_clad_middle_z,
     th_system.thermal_state.T_clad_in_z, th_system.thermal_state.k_clad_out,
     th_system.thermal_state.k_clad_mid, th_system.thermal_state.k_clad_in) = converge_cladding(th_system, tolerance, max_iterations)

    (th_system.thermal_state.T_fuel_avg_z, th_system.thermal_state.T_fuel_y,
     th_system.thermal_state.k_fuel) = converge_fuel(th_system, tolerance, max_iterations)

    # Update final state
    update_final_thermal_state(th_system)

    # Calculate critical heat flux and MDNBR using the main system's method
    th_system.calculate_critical_heat_flux()

    return th_system.thermal_state
