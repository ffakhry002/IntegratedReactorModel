import numpy as np
import os
from CoolProp.CoolProp import PropsSI

def get_coolant_properties(th_system, temp_vector):
    """Get coolant properties based on type and temperature.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing coolant type and other properties
    temp_vector : numpy.ndarray
        Array of temperatures to get properties for

    Returns
    -------
    tuple
        (coolant_density, specific_heat_capacity, thermal_conductivity, viscosity)
    """
    # Determine coolant fluid name for CoolProp
    fluid = 'Water' if th_system.material.coolant_type == 'Light Water' else 'HeavyWater'

    # Get pressure in Pa (default to 1 atm if not specified)
    pressure = getattr(th_system.thermal_hydraulics, 'reactor_pressure', 101325.0)

    # Calculate properties using CoolProp
    coolant_density = np.zeros_like(temp_vector)
    specific_heat_capacity = np.zeros_like(temp_vector)
    thermal_conductivity = np.zeros_like(temp_vector)
    viscosity = np.zeros_like(temp_vector)

    for i, temp in enumerate(temp_vector):
        coolant_density[i] = PropsSI('D', 'T', temp, 'P', pressure, fluid)  # kg/m^3
        specific_heat_capacity[i] = PropsSI('C', 'T', temp, 'P', pressure, fluid)  # J/kg-K
        thermal_conductivity[i] = PropsSI('L', 'T', temp, 'P', pressure, fluid)  # W/m-K
        viscosity[i] = PropsSI('V', 'T', temp, 'P', pressure, fluid)  # Pa-s

    return coolant_density, specific_heat_capacity, thermal_conductivity, viscosity

def calculate_heat_transfer_coeff_coolant(th_system):
    """Calculate coolant heat transfer coefficient using appropriate correlation.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing all necessary properties

    Returns
    -------
    float
        Heat transfer coefficient in W/m^2-K
    """
    # Get properties from thermal state
    coolant_density = th_system.thermal_state.coolant_density
    specific_heat_capacity = th_system.thermal_state.specific_heat_capacity
    thermal_conductivity = th_system.thermal_state.thermal_conductivity
    viscosity = th_system.thermal_state.viscosity

    # Calculate dimensionless numbers
    Pr = specific_heat_capacity * viscosity / thermal_conductivity
    Re = coolant_density * th_system.thermal_hydraulics.flow_rate * th_system.geometry.hydraulic_diameter / viscosity

    if th_system.thermal_hydraulics.assembly_type == 'Plate':
        return 0.036 * Re**0.76 * Pr**(1/3) * thermal_conductivity / th_system.geometry.hydraulic_diameter
    else:  # Pin geometry
        return 0.023 * Re**0.8 * Pr**0.4 * thermal_conductivity / th_system.geometry.hydraulic_diameter

def calculate_mass_flow_rate(th_system):
    """Calculate mass flow rate.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing all necessary properties

    Returns
    -------
    float
        Mass flow rate in kg/s
    """
    return (th_system.geometry.coolant_area *
            np.mean(th_system.thermal_state.coolant_density) *
            th_system.thermal_hydraulics.flow_rate)

def get_saturated_values(th_system):
    """Get saturated properties for the coolant.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing reactor pressure information

    Returns
    -------
    tuple
        (T_sat, h_fg, mu_f, Cp_f)
        - T_sat: Saturation temperature (K)
        - h_fg: Latent heat of vaporization (J/kg)
        - mu_f: Dynamic viscosity (Pa-s)
        - Cp_f: Specific heat capacity (J/kg-K)
    """
    pressure = th_system.thermal_hydraulics.reactor_pressure
    fluid = 'Water' if getattr(th_system.material, 'coolant_type', 'Light Water') == 'Light Water' else 'HeavyWater'

    # Saturation temperature at the given pressure
    T_sat = PropsSI('T', 'P', pressure, 'Q', 0, fluid)  # K

    # Latent heat of vaporization
    h_f = PropsSI('H', 'P', pressure, 'Q', 0, fluid)  # J/kg (saturated liquid)
    h_g = PropsSI('H', 'P', pressure, 'Q', 1, fluid)  # J/kg (saturated vapor)
    h_fg = h_g - h_f  # J/kg

    # Dynamic viscosity of saturated liquid
    mu_f = PropsSI('V', 'P', pressure, 'Q', 0, fluid)  # Pa-s

    # Specific heat capacity of saturated liquid
    Cp_f = PropsSI('C', 'P', pressure, 'Q', 0, fluid)  # J/kg-K

    return T_sat, h_fg, mu_f, Cp_f
