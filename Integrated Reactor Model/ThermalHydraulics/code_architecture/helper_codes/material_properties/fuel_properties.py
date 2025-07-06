import numpy as np
import math

def calculate_k_fuel(th_system, fuel_temp):
    """Calculate fuel thermal conductivity based on type and temperature.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing material information
    fuel_temp : float
        Temperature at which to calculate conductivity

    Returns
    -------
    float
        Thermal conductivity in W/m-K
    """
    # Convert temperature to float to avoid integer math issues
    fuel_temp = float(fuel_temp)
    fuel_type = th_system.material.fuel_type

    if fuel_type == 'U3Si2':
        T1, k1 = 300.0, 10.0
        T2, k2 = 1773.0, 32.0
        return k1 + (k2 - k1) * (fuel_temp - T1) / (T2 - T1)
    elif fuel_type == 'U10Mo':
        return 10.2 + (3.51*(10.0**(-2))) * (fuel_temp-273.15)
    elif fuel_type == 'UO2':
        A1, A2, A3, A4 = 0.0375, 2.165e-4, 1.70e-3, 0.058
        Bu = 0.0  # GWd/Mt_IHM
        f = (1.0 + math.exp((fuel_temp - 900.0) / 80.0)) ** (-1.0)
        g = 4.715e9 * (fuel_temp ** (-2.0)) * math.exp(-16361.0 / fuel_temp)
        return 1.0 / (A1 + A2 * fuel_temp + A3 * Bu + A4 * f) + g

def calculate_k_fuel_vector(th_system, T_fuel_array):
    """Vectorized calculation of fuel thermal conductivity.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing material information
    T_fuel_array : numpy.ndarray
        Array of temperatures

    Returns
    -------
    numpy.ndarray
        Array of thermal conductivities in W/m-K
    """
    # Convert input array to float type
    T_fuel_array = T_fuel_array.astype(np.float64)
    return np.vectorize(lambda T: calculate_k_fuel(th_system, T))(T_fuel_array)
