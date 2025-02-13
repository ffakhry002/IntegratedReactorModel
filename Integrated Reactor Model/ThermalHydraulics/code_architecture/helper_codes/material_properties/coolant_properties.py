import numpy as np
import pandas as pd
import os

def get_coolant_properties(th_system, temp_vector):
    """Get coolant properties based on type and temperature.

    Args:
        th_system: THSystem object containing coolant type and other properties
        temp_vector (np.array): Array of temperatures to get properties for

    Returns:
        tuple: (coolant_density, specific_heat_capacity, thermal_conductivity, viscosity)
    """
    # Get the path to the coolant data files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = 'isobaric_lightwater_1atm.csv' if th_system.material.coolant_type == 'Light Water' else 'isobaric_heavywater_1atm.csv'
    data_path = os.path.join(current_dir, 'coolant_data', data_file)

    # Read and process the data
    data = pd.read_csv(data_path)
    data['Temperature [K]'] = data['Temperature [K]'].astype(float)

    # Interpolate properties
    coolant_density = np.interp(temp_vector, data['Temperature [K]'], data['Density [g/ml]']) * 1000
    specific_heat_capacity = np.interp(temp_vector, data['Temperature [K]'], data['Specific Heat Capacity [J/kg.K]'])
    thermal_conductivity = np.interp(temp_vector, data['Temperature [K]'], data['Thermal Conductivity [W/m.K]'])
    viscosity = np.interp(temp_vector, data['Temperature [K]'], data['Dynamic Viscosity [Pa.s]'])

    return coolant_density, specific_heat_capacity, thermal_conductivity, viscosity

def calculate_heat_transfer_coeff_coolant(th_system):
    """Calculate coolant heat transfer coefficient using appropriate correlation.

    Args:
        th_system: THSystem object containing all necessary properties

    Returns:
        float: Heat transfer coefficient in W/m^2-K
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

    Args:
        th_system: THSystem object containing all necessary properties

    Returns:
        float: Mass flow rate in kg/s
    """
    return (th_system.geometry.coolant_area *
            np.mean(th_system.thermal_state.coolant_density) *
            th_system.thermal_hydraulics.flow_rate)

def get_saturated_values(th_system):
    """Get saturated properties for the coolant.

    Args:
        th_system: THSystem object containing reactor pressure information

    Returns:
        tuple: (T_sat, h_fg, mu_f, Cp_f)
        - T_sat: Saturation temperature (K)
        - h_fg: Latent heat of vaporization (J/kg)
        - mu_f: Dynamic viscosity (Pa-s)
        - Cp_f: Specific heat capacity (J/kg-K)
    """
    def Ts(P):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                        210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        Psat = np.array([0.006112, 0.012271, 0.023368, 0.042418, 0.073750, 0.12335, 0.19919, 0.31161, 0.47358, 0.70109, 1.01325,
                        1.4327, 1.9854, 2.7011, 3.6136, 4.7597, 6.1804, 7.9202, 10.027, 12.553, 15.550, 19.080, 23.202, 27.979,
                        33.480, 39.776, 46.941, 55.052, 64.191, 74.449, 85.917, 98.694, 112.89, 128.64, 146.08, 165.37, 186.74,
                        210.53, 221.2]) * 1e5  # Convert to Pa
        return np.interp(th_system.thermal_hydraulics.reactor_pressure, Psat, Tsat)

    def hfg(T):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                        210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        deltah = np.array([2501, 2477, 2454, 2430, 2406, 2382, 2358, 2333, 2308, 2283, 2257, 2230, 2203, 2174, 2145, 2114, 2083,
                          2050, 2015, 1979, 1941, 1900, 1857, 1811, 1762, 1709, 1652, 1590, 1523, 1450, 1370, 1282, 1184, 1074,
                          948, 801, 623, 374, 0]) * 1000  # Convert to J/kg
        return np.interp(T, Tsat, deltah)

    def muf(T):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                        210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        muliq = np.array([1786, 1304, 1002, 798.3, 653.9, 547.8, 467.3, 404.8, 355.4, 315.6, 283.1, 254.8, 231.0, 210.9, 194.1, 179.8,
                         167.7, 157.4, 148.5, 140.7, 133.9, 127.9, 122.4, 117.5, 112.9, 108.7, 104.8, 101.1, 97.5, 94.1, 90.7,
                         87.2, 83.5, 79.5, 75.4, 69.4, 62.1, 51.8, 41.4]) * 1e-6  # Already in Pa*s
        return np.interp(T, Tsat, muliq)

    def Cpf(T):
        Tsat = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                        210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 374.15]) + 273.15
        Cpliq = np.array([4.218, 4.194, 4.182, 4.179, 4.179, 4.181, 4.185, 4.191, 4.198, 4.207, 4.218, 4.230, 4.244, 4.262,
                         4.282, 4.306, 4.334, 4.366, 4.403, 4.446, 4.494, 4.550, 4.613, 4.685, 4.769, 4.866, 4.985, 5.134,
                         5.307, 5.520, 5.794, 6.143, 6.604, 7.241, 8.225, 10.07, 15, 55, 1e6]) * 1000  # Convert to J/(kg*K)
        return np.interp(T, Tsat, Cpliq)

    T_sat = Ts(th_system.thermal_hydraulics.reactor_pressure)
    h_fg = hfg(T_sat)
    mu_f = muf(T_sat)
    Cp_f = Cpf(T_sat)

    return T_sat, h_fg, mu_f, Cp_f
