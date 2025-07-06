import numpy as np
from tabulate import tabulate

def converge_coolant(th_system, tolerance=0.001, max_iterations=100):
    """Iteratively converge coolant temperatures until convergence criteria is met.

    Parameters
    ----------
    th_system : THSystem
        THSystem object containing geometry and material information
    tolerance : float, optional
        Maximum allowed temperature difference between iterations. Defaults to 0.001.
    max_iterations : int, optional
        Maximum number of iterations to attempt. Defaults to 100.

    Returns
    -------
    tuple
        Contains:
            - np.array: Converged coolant temperatures along z-axis
            - float: Average coolant temperature

    Notes
    -----
    Prints convergence information including iteration count, temperature statistics,
    and convergence status.
    """
    iteration = 0
    print("Starting coolant convergence process:")
    headers = ["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Max Diff (K)"]
    rows = [headers]

    while iteration < max_iterations:
        new_T_coolant_z, *_ = th_system.single_iteration()
        max_diff = np.max(np.abs(new_T_coolant_z - th_system.thermal_state.T_coolant_z))

        row = [iteration + 1, f"{np.min(new_T_coolant_z):.2f}", f"{np.max(new_T_coolant_z):.2f}",
               f"{np.mean(new_T_coolant_z):.2f}", f"{max_diff:.6f}"]
        rows.append(row)

        if max_diff < tolerance:
            print(tabulate(rows, headers="firstrow", tablefmt="grid"))
            print(f"\nCoolant converged after {iteration + 1} iterations")
            return new_T_coolant_z, np.mean(new_T_coolant_z)

        th_system.thermal_state.T_coolant_z = new_T_coolant_z
        iteration += 1

    print(tabulate(rows, headers="firstrow", tablefmt="grid"))
    print(f"\nWarning: Coolant did not converge after {max_iterations} iterations")
    return th_system.thermal_state.T_coolant_z, np.mean(th_system.thermal_state.T_coolant_z)
