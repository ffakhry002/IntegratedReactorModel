import numpy as np
from tabulate import tabulate
from ..material_properties.clad_properties import calculate_k_clad, calculate_k_clad_vector

def print_cladding_convergence_tables(rows_out, rows_mid, rows_in, iteration):
    """Helper method to print cladding convergence tables."""
    print("Outer Cladding Convergence:")
    print(tabulate(rows_out, headers="firstrow", tablefmt="grid"))
    print("\nMiddle Cladding Convergence:")
    print(tabulate(rows_mid, headers="firstrow", tablefmt="grid"))
    print("\nInner Cladding Convergence:")
    print(tabulate(rows_in, headers="firstrow", tablefmt="grid"))
    if iteration >= 100:
        print(f"\nWarning: Cladding did not converge after {iteration} iterations")
    else:
        print(f"\nCladding converged after {iteration + 1} iterations")

def converge_cladding(th_system, tolerance=0.001, max_iterations=100):
    """Converge cladding temperatures and conductivities through iteration.

    Args:
        th_system: THSystem object containing geometry and material information
        tolerance (float, optional): Convergence tolerance. Defaults to 0.001.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: Contains the following arrays:
            - T_clad_out_z: Outer cladding temperatures
            - T_clad_middle_z: Middle cladding temperatures
            - T_clad_in_z: Inner cladding temperatures
            - new_k_clad_out: Outer cladding conductivities
            - new_k_clad_mid: Middle cladding conductivities
            - new_k_clad_in: Inner cladding conductivities
    """
    iteration = 0
    print("\nStarting cladding convergence process:")
    rows_out = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Coeff (W/m-K)", "Max Temp Diff (K)"]]
    rows_mid = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Coeff (W/m-K)", "Max Temp Diff (K)"]]
    rows_in = [["Iteration", "Min Temp (K)", "Max Temp (K)", "Avg Temp (K)", "Coeff (W/m-K)", "Max Temp Diff (K)"]]

    while iteration < max_iterations:
        _, T_clad_out_z, T_clad_middle_z, T_clad_in_z, *_ = th_system.single_iteration()

        new_k_clad_out = calculate_k_clad_vector(th_system, T_clad_out_z)
        new_k_clad_mid = calculate_k_clad_vector(th_system, T_clad_middle_z)
        new_k_clad_in = calculate_k_clad_vector(th_system, T_clad_in_z)

        max_temp_diff_out = np.max(np.abs(new_k_clad_out - th_system.thermal_state.k_clad_out))
        max_temp_diff_mid = np.max(np.abs(new_k_clad_mid - th_system.thermal_state.k_clad_mid))
        max_temp_diff_in = np.max(np.abs(new_k_clad_in - th_system.thermal_state.k_clad_in))

        rows_out.append([iteration + 1, f"{np.min(T_clad_out_z):.2f}", f"{np.max(T_clad_out_z):.2f}",
                        f"{np.mean(T_clad_out_z):.2f}", f"{np.mean(new_k_clad_out):.2f}", f"{max_temp_diff_out:.6f}"])
        rows_mid.append([iteration + 1, f"{np.min(T_clad_middle_z):.2f}", f"{np.max(T_clad_middle_z):.2f}",
                        f"{np.mean(T_clad_middle_z):.2f}", f"{np.mean(new_k_clad_mid):.2f}", f"{max_temp_diff_mid:.6f}"])
        rows_in.append([iteration + 1, f"{np.min(T_clad_in_z):.2f}", f"{np.max(T_clad_in_z):.2f}",
                       f"{np.mean(T_clad_in_z):.2f}", f"{np.mean(new_k_clad_in):.2f}", f"{max_temp_diff_in:.6f}"])

        if max(max_temp_diff_out, max_temp_diff_mid, max_temp_diff_in) < tolerance:
            print_cladding_convergence_tables(rows_out, rows_mid, rows_in, iteration)
            return T_clad_out_z, T_clad_middle_z, T_clad_in_z, new_k_clad_out, new_k_clad_mid, new_k_clad_in

        th_system.thermal_state.k_clad_out = new_k_clad_out
        th_system.thermal_state.k_clad_mid = new_k_clad_mid
        th_system.thermal_state.k_clad_in = new_k_clad_in
        iteration += 1

    print_cladding_convergence_tables(rows_out, rows_mid, rows_in, iteration)
    return T_clad_out_z, T_clad_middle_z, T_clad_in_z, th_system.thermal_state.k_clad_out, th_system.thermal_state.k_clad_mid, th_system.thermal_state.k_clad_in
