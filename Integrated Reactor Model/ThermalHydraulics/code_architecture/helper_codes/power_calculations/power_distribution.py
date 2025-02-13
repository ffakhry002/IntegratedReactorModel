import numpy as np
from scipy.integrate import trapezoid

def calculate_Q_dot_z(th_system):
    """Calculate axial power distribution.

    Args:
        th_system: THSystem object containing geometry, power, and material information

    Returns:
        np.array: Array of heat generation rates along z-axis
    """
    # Calculate assembly volumes
    if th_system.thermal_hydraulics.assembly_type == 'Plate':
        single_assembly_volume = (
            (th_system.plate_geometry.fuel_plate_width + 2 * th_system.plate_geometry.clad_structure_width) *
            th_system.plate_geometry.fuel_plate_pitch *
            th_system.plate_geometry.fuel_height *
            th_system.plate_geometry.plates_per_assembly
        )
    else:
        single_assembly_volume = (
            (th_system.pin_geometry.pin_pitch * th_system.pin_geometry.n_side_pins)**2 *
            th_system.pin_geometry.fuel_height
        )

    total_assembly_volume = single_assembly_volume * th_system.reactor_power.num_assemblies

    # Calculate power density
    if th_system.reactor_power.Q_dot_z_toggle == 'PD':
        avg_power_density_kW_L = th_system.reactor_power.input_avg_power_density
        avg_power_density_W_m3 = avg_power_density_kW_L * 1e6
    else:
        avg_power_density_kW_L = th_system.reactor_power.core_power / total_assembly_volume
        avg_power_density_W_m3 = avg_power_density_kW_L * 1e6

    # Calculate power per element
    total_power_per_assembly = avg_power_density_W_m3 * single_assembly_volume
    power_per_element = total_power_per_assembly / th_system.geometry.n_elements_per_assembly

    # Calculate linear power based on toggle
    if th_system.reactor_power.Q_dot_z_toggle in ['CP', 'PD']:
        Average_linear_power = power_per_element / th_system.geometry.fuel_height
        Peak_linear_power = Average_linear_power * np.pi / 2
    elif th_system.reactor_power.Q_dot_z_toggle == 'ALP':
        Average_linear_power = th_system.reactor_power.input_avg_lp * 1e3
        Peak_linear_power = Average_linear_power * np.pi / 2
    elif th_system.reactor_power.Q_dot_z_toggle == 'MLP':
        Peak_linear_power = th_system.reactor_power.input_max_lp * 1e3
        Average_linear_power = Peak_linear_power * 2 / np.pi

    # Calculate power distribution
    original_curve = np.cos(np.pi * th_system.z / th_system.geometry.fuel_height)
    original_power = trapezoid(original_curve, th_system.z)

    adjusted_curve = (1 - th_system.reactor_power.cos_curve_squeeze) * original_curve + th_system.reactor_power.cos_curve_squeeze
    new_power = trapezoid(adjusted_curve, th_system.z)
    adjusted_curve *= original_power / new_power

    # Calculate final Q_dot_z
    if th_system.reactor_power.Q_dot_z_toggle == 'MLP':
        adjusted_curve = adjusted_curve / np.max(adjusted_curve)
        Q_dot_z = Peak_linear_power * adjusted_curve
    else:
        Q_dot_z = Peak_linear_power * adjusted_curve

    return Q_dot_z
