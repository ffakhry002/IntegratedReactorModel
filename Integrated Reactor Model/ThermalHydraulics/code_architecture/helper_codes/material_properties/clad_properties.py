import numpy as np

def calculate_k_clad(th_system, clad_temp):
    """Calculate cladding thermal conductivity based on type and temperature.

    Args:
        th_system: THSystem object containing material information
        clad_temp (float): Temperature at which to calculate conductivity

    Returns:
        float: Thermal conductivity in W/m-K
    """
    clad_type = th_system.material.clad_type

    if clad_type == 'Al6061':
        data_points = [
            (4, 5.347), (50, 62.05), (100, 97.7),
            (150, 120.4), (200, 136), (300, 156.3),
            (400, 171.6), (500, 176.8), (600, 179.3),
            (673, 180)
        ]
        data_points.sort(key=lambda x: x[0])

        if clad_temp <= data_points[0][0]:
            return data_points[0][1]
        elif clad_temp >= data_points[-1][0]:
            return data_points[-1][1]

        for i in range(len(data_points) - 1):
            T1, k1 = data_points[i]
            T2, k2 = data_points[i + 1]
            if T1 <= clad_temp <= T2:
                slope = (k2 - k1) / (T2 - T1)
                return k1 + slope * (clad_temp - T1)

    elif clad_type == 'Zirc2':
        k = 0.138 - 3.90e-5 * clad_temp + 1.184e-7 * clad_temp ** 2
        return k * 100  # Convert to W/m-K
    elif clad_type == 'Zirc4':
        k = 0.113 + 2.25e-5 * clad_temp + 0.725e-8 * clad_temp ** 2
        return k * 100  # Convert to W/m-K

def calculate_k_clad_vector(th_system, T_clad_array):
    """Vectorized calculation of cladding thermal conductivity.

    Args:
        th_system: THSystem object containing material information
        T_clad_array (np.array): Array of temperatures to calculate conductivity for

    Returns:
        np.array: Array of thermal conductivities in W/m-K, corresponding to input temperatures
    """
    return np.vectorize(lambda T: calculate_k_clad(th_system, T))(T_clad_array)
