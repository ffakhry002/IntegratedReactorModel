import numpy as np
from ..material_properties.coolant_properties import get_saturated_values

def calculate_q_dnb_vector(th_system, T_coolant_z):
    """Calculate departure from nucleate boiling ratio.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing geometry and material information
    T_coolant_z : numpy.ndarray
        Array of coolant temperatures

    Returns
    -------
    numpy.ndarray
        Array of critical heat fluxes
    """
    # Get saturated properties
    T_sat, h_fg, mu_f, Cp_f = get_saturated_values(th_system)

    # Calculate thermodynamic quality
    x_e = - Cp_f * (T_sat - T_coolant_z) / h_fg  # Dimensionless

    # Calculate mass flux
    G = th_system.thermal_state.mass_flow_rate / th_system.geometry.coolant_area

    # Convert pressure to MPa
    P_MPa = th_system.thermal_hydraulics.reactor_pressure * 1e-6

    # Calculate Tong factor
    K_tong = (1.76 - 7.433*x_e + 12.222*(x_e**2)) * (
        1 - (52.3 + 80*x_e - 50*(x_e**2)) / (60.5 + (10*P_MPa)**1.4))

    # Calculate critical heat flux
    q_dnb = K_tong * (G**0.4) * (mu_f**0.6) * h_fg / (th_system.geometry.hydraulic_diameter**0.6)
    return q_dnb

def calculate_heat_flux_z(th_system):
    """Calculate heat flux along z-axis.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing geometry and thermal state

    Returns
    -------
    numpy.ndarray
        Array of heat fluxes
    """
    if th_system.thermal_hydraulics.assembly_type == 'Pin':
        return th_system.thermal_state.Q_dot_z / (2 * np.pi * th_system.pin_geometry.r_clad_outer)
    else:
        return th_system.thermal_state.Q_dot_z / (2 * th_system.plate_geometry.fuel_plate_width)

def calculate_critical_heat_flux(th_system):
    """Calculate critical heat flux and MDNBR.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing geometry and thermal state

    Returns
    -------
    tuple
        Arrays of critical heat flux, heat flux, and MDNBR
    """
    # Calculate DNB heat flux
    q_dnb = calculate_q_dnb_vector(th_system, th_system.thermal_state.T_coolant_z)

    # Calculate actual heat flux
    heat_flux_z = calculate_heat_flux_z(th_system)

    # Calculate MDNBR
    MDNBR = q_dnb / heat_flux_z

    return q_dnb, heat_flux_z, MDNBR
