import numpy as np
from tabulate import tabulate
from ..material_properties.fuel_properties import calculate_k_fuel_vector

def converge_fuel(th_system, tolerance=0.001, max_iterations=100, relaxation_factor=0.3):
    """Iteratively converge fuel temperatures and conductivities with relaxation.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing geometry and material information
    tolerance : float, optional
        Maximum allowed conductivity difference between iterations
    max_iterations : int, optional
        Maximum number of iterations to attempt
    relaxation_factor : float, optional
        Factor to dampen changes between iterations (0 to 1)

    Returns
    -------
    tuple
        Contains:
            - np.array: Average fuel temperatures along z-axis
            - np.array: 2D array of fuel temperatures (z, y)
            - np.array: Converged fuel thermal conductivities

    Notes
    -----
    Prints convergence information including iteration count, temperature statistics,
    conductivity values, and convergence status.
    """
    iteration = 0
    print("\nStarting fuel convergence process:")
    rows = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Avg Coeff (W/m-K)", "Max Temp Diff (K)"]]

    while iteration < max_iterations:
        _, _, _, _, _, _, T_fuel_avg, T_fuel_y, _, _, _, _ = th_system.single_iteration()

        new_k_fuel = calculate_k_fuel_vector(th_system, T_fuel_y)

        # Apply relaxation to the change in conductivity
        relaxed_k_fuel = (
            th_system.thermal_state.k_fuel * (1 - relaxation_factor) +
            new_k_fuel * relaxation_factor
        )

        max_temp_diff = np.max(np.abs(relaxed_k_fuel - th_system.thermal_state.k_fuel))
        row = [iteration + 1, f"{np.min(T_fuel_y):.2f}", f"{np.max(T_fuel_y):.2f}",
               f"{np.mean(T_fuel_avg):.2f}", f"{np.mean(relaxed_k_fuel):.2f}", f"{max_temp_diff:.6f}"]
        rows.append(row)

        if max_temp_diff < tolerance:
            print(tabulate(rows, headers="firstrow", tablefmt="grid"))
            print(f"\nFuel converged after {iteration + 1} iterations")
            return T_fuel_avg, T_fuel_y, relaxed_k_fuel

        th_system.thermal_state.k_fuel = relaxed_k_fuel
        iteration += 1

    print(tabulate(rows, headers="firstrow", tablefmt="grid"))
    print(f"\nWarning: Fuel did not converge after {max_iterations} iterations")
    return T_fuel_avg, T_fuel_y, th_system.thermal_state.k_fuel
